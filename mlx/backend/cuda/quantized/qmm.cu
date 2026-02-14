// Copyright © 2025 Apple Inc.
//
// Fused Quantized Matrix-Matrix Multiplication (fp_qmm)
// Performs: out = x @ dequantize(w, scales).T  (when transpose=true)
//           out = x @ dequantize(w, scales)    (when transpose=false)
//

#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm.cuh"
#include "mlx/backend/cuda/quantized/quantized_utils.cuh"
#include "mlx/backend/cuda/steel/defines.cuh"
#include "mlx/backend/cuda/steel/utils.cuh"
#include "mlx/dtype_utils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace mlx::core::cu {

// =============================================================================
// Tile sizes for tensor core GEMM (128x128 tiles for better arithmetic intensity)
// =============================================================================
static constexpr int TC_BM = 128;  // Rows of output tile (M dimension)
static constexpr int TC_BN = 128;  // Cols of output tile (N dimension)
static constexpr int TC_BK = 32;   // Reduction dimension tile (K dimension)

// Thread block configuration: 8 warps = 256 threads (for 128x128 tiles)
static constexpr int TC_WARPS_M = 4;
static constexpr int TC_WARPS_N = 2;
static constexpr int TC_NUM_WARPS = TC_WARPS_M * TC_WARPS_N;
static constexpr int TC_BLOCK_SIZE = TC_NUM_WARPS * 32;

// Each warp handles this much of the output tile
static constexpr int TC_WARP_M = TC_BM / TC_WARPS_M;  // 32
static constexpr int TC_WARP_N = TC_BN / TC_WARPS_N;  // 64

// =============================================================================
// Dequantization helpers
// =============================================================================

// Dequantize FP8 packed values to float4
__device__ __forceinline__ float4 dequant_fp8_scaled(uint32_t bits, float scale) {
  auto out = *(__nv_fp8x4_e4m3*)(&bits);
  float4 f = out.operator float4();
  return make_float4(f.x * scale, f.y * scale, f.z * scale, f.w * scale);
}

// Dequantize FP4 packed values to float4
__device__ __forceinline__ float4 dequant_fp4_scaled(uint16_t bits, float scale) {
  auto out = *(__nv_fp4x4_e2m1*)(&bits);
  float4 f = out.operator float4();
  return make_float4(f.x * scale, f.y * scale, f.z * scale, f.w * scale);
}

// Load scale factor and convert to float
template <bool use_mx_scale>
__device__ __forceinline__ float load_scale(const uint8_t* scales, int idx) {
  uint8_t s = scales[idx];
  if constexpr (use_mx_scale) {
    return float(*(__nv_fp8_e8m0*)(&s));
  } else {
    return float(*(__nv_fp8_e4m3*)(&s));
  }
}

// =============================================================================
// Tensor Core 16x16 Tile (distributed across warp)
// =============================================================================
struct Tile16x16_bf16 {
  __nv_bfloat162 values[4];  // Each thread holds 8 bf16 values as 4 bf16x2

  // Load 16x16 tile from shared memory using ldmatrix
  __device__ __forceinline__ void load(uint32_t row_address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(*(uint32_t*)&(values[0])),
          "=r"(*(uint32_t*)&(values[1])),
          "=r"(*(uint32_t*)&(values[2])),
          "=r"(*(uint32_t*)&(values[3]))
        : "r"(row_address));
  }
};

struct Tile16x16_f32 {
  float2 values[4];  // Each thread holds 8 float values as 4 float2

  __device__ inline void fill(float v) {
    float2 v2 = {v, v};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      values[i] = v2;
    }
  }

  // Store to global memory
  template <typename T>
  __device__ inline void store_global(T* x, int N) {
    const int laneid = threadIdx.x % 32;
    const int row = laneid / 4;
    const int col = laneid % 4;

    // Fragment layout: each thread has values at specific row/col positions
    x[(row + 0) * N + 2 * col + 0] = T(values[0].x);
    x[(row + 0) * N + 2 * col + 1] = T(values[0].y);
    x[(row + 0) * N + 2 * col + 8] = T(values[2].x);
    x[(row + 0) * N + 2 * col + 9] = T(values[2].y);
    x[(row + 8) * N + 2 * col + 0] = T(values[1].x);
    x[(row + 8) * N + 2 * col + 1] = T(values[1].y);
    x[(row + 8) * N + 2 * col + 8] = T(values[3].x);
    x[(row + 8) * N + 2 * col + 9] = T(values[3].y);
  }

  // Bounds-checked store for partial tiles
  template <typename T>
  __device__ inline void store_global_safe(T* x, int N, int max_rows, int max_cols) {
    const int laneid = threadIdx.x % 32;
    const int row = laneid / 4;
    const int col = laneid % 4;

    if (row < max_rows) {
      if (2 * col + 0 < max_cols) x[(row + 0) * N + 2 * col + 0] = T(values[0].x);
      if (2 * col + 1 < max_cols) x[(row + 0) * N + 2 * col + 1] = T(values[0].y);
      if (2 * col + 8 < max_cols) x[(row + 0) * N + 2 * col + 8] = T(values[2].x);
      if (2 * col + 9 < max_cols) x[(row + 0) * N + 2 * col + 9] = T(values[2].y);
    }
    if (row + 8 < max_rows) {
      if (2 * col + 0 < max_cols) x[(row + 8) * N + 2 * col + 0] = T(values[1].x);
      if (2 * col + 1 < max_cols) x[(row + 8) * N + 2 * col + 1] = T(values[1].y);
      if (2 * col + 8 < max_cols) x[(row + 8) * N + 2 * col + 8] = T(values[3].x);
      if (2 * col + 9 < max_cols) x[(row + 8) * N + 2 * col + 9] = T(values[3].y);
    }
  }
};

// =============================================================================
// Tensor Core MMA operation: C += A @ B.T
// Uses mma.sync.aligned.m16n8k16 instruction
// =============================================================================
__device__ __forceinline__ void mma_bf16(
    Tile16x16_f32& C,
    Tile16x16_bf16& A,
    Tile16x16_bf16& B) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  // First m16n8k16 operation (computes left half of 16x16 C tile)
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};"
      : "+f"(C.values[0].x), "+f"(C.values[0].y),
        "+f"(C.values[1].x), "+f"(C.values[1].y)
      : "r"(*(uint32_t*)(&A.values[0])),
        "r"(*(uint32_t*)(&A.values[1])),
        "r"(*(uint32_t*)(&A.values[2])),
        "r"(*(uint32_t*)(&A.values[3])),
        "r"(*(uint32_t*)(&B.values[0])),
        "r"(*(uint32_t*)(&B.values[2])),
        "f"(C.values[0].x), "f"(C.values[0].y),
        "f"(C.values[1].x), "f"(C.values[1].y));

  // Second m16n8k16 operation (computes right half of 16x16 C tile)
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};"
      : "+f"(C.values[2].x), "+f"(C.values[2].y),
        "+f"(C.values[3].x), "+f"(C.values[3].y)
      : "r"(*(uint32_t*)(&A.values[0])),
        "r"(*(uint32_t*)(&A.values[1])),
        "r"(*(uint32_t*)(&A.values[2])),
        "r"(*(uint32_t*)(&A.values[3])),
        "r"(*(uint32_t*)(&B.values[1])),
        "r"(*(uint32_t*)(&B.values[3])),
        "f"(C.values[2].x), "f"(C.values[2].y),
        "f"(C.values[3].x), "f"(C.values[3].y));
#endif
}

// =============================================================================
// Tensor Core Quantized GEMM Kernel
// Computes: out[M,N] = x[M,K] @ dequant(w[N,K/pack]).T
// =============================================================================
template <typename T, int bits, int group_size, bool use_mx_scale>
__global__ void fp_qmm_tensor_core_kernel(
    const T* __restrict__ x,              // [M, K] input activations
    const uint32_t* __restrict__ w,       // [N, K/pack_factor] quantized weights
    const uint8_t* __restrict__ scales,   // [N, K/group_size] scale factors
    T* __restrict__ out,                  // [M, N] output
    int M, int N, int K) {

  constexpr int pack_factor = 32 / bits;  // 4 for fp8, 8 for fp4
  constexpr int vals_per_pack = (bits == 8) ? 4 : 8;

  // Thread/warp indices
  const int warpid = threadIdx.x / 32;
  const int laneid = threadIdx.x % 32;
  const int wm = warpid / TC_WARPS_N;
  const int wn = warpid % TC_WARPS_N;

  // This warp's position in the output tile
  const int warp_offset_m = wm * TC_WARP_M;
  const int warp_offset_n = wn * TC_WARP_N;

  // Block position in output
  const int bm = blockIdx.y * TC_BM;
  const int bn = blockIdx.x * TC_BN;

  // Precompute constants
  const int K_packed = K / pack_factor;
  const int K_groups = K / group_size;
  const int num_k_tiles = (K + TC_BK - 1) / TC_BK;

  // Simple shared memory layout (no swizzling for simplicity)
  extern __shared__ char shmem[];
  __nv_bfloat16* x_tile = (__nv_bfloat16*)shmem;
  __nv_bfloat16* w_tile = x_tile + TC_BM * TC_BK;

  // Accumulator tiles (2x4 = 8 tiles of 16x16 for 32x64 output per warp)
  Tile16x16_f32 C[8];
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    C[i].fill(0.0f);
  }

  // ==========================================================================
  // Main K-tile loop
  // ==========================================================================
  for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
    const int k_base = k_tile * TC_BK;

    // Load x tile [TC_BM, TC_BK] - coalesced access
    for (int i = threadIdx.x; i < TC_BM * TC_BK; i += TC_BLOCK_SIZE) {
      const int row = i / TC_BK;
      const int col = i % TC_BK;
      const int global_row = bm + row;
      const int global_col = k_base + col;

      __nv_bfloat16 val;
      if (global_row < M && global_col < K) {
        val = __float2bfloat16(float(x[global_row * K + global_col]));
      } else {
        val = __float2bfloat16(0.0f);
      }
      x_tile[row * TC_BK + col] = val;
    }

    // Load and dequantize w tile [TC_BN, TC_BK]
    for (int i = threadIdx.x; i < (TC_BN * TC_BK) / vals_per_pack; i += TC_BLOCK_SIZE) {
      const int linear_idx = i * vals_per_pack;
      const int n_local = linear_idx / TC_BK;
      const int k_local = linear_idx % TC_BK;

      const int global_n = bn + n_local;
      const int global_k = k_base + k_local;

      if (global_n < N && global_k < K) {
        const int pack_idx = global_k / pack_factor;
        const int scale_idx = global_k / group_size;

        uint32_t packed = w[global_n * K_packed + pack_idx];
        float scale = load_scale<use_mx_scale>(scales + global_n * K_groups, scale_idx);

        if constexpr (bits == 8) {
          float4 vals = dequant_fp8_scaled(packed, scale);
          w_tile[n_local * TC_BK + k_local + 0] = __float2bfloat16(vals.x);
          w_tile[n_local * TC_BK + k_local + 1] = __float2bfloat16(vals.y);
          w_tile[n_local * TC_BK + k_local + 2] = __float2bfloat16(vals.z);
          w_tile[n_local * TC_BK + k_local + 3] = __float2bfloat16(vals.w);
        } else {
          uint16_t lo = packed & 0xFFFF;
          uint16_t hi = (packed >> 16) & 0xFFFF;
          float4 vals_lo = dequant_fp4_scaled(lo, scale);
          float4 vals_hi = dequant_fp4_scaled(hi, scale);

          w_tile[n_local * TC_BK + k_local + 0] = __float2bfloat16(vals_lo.x);
          w_tile[n_local * TC_BK + k_local + 1] = __float2bfloat16(vals_lo.y);
          w_tile[n_local * TC_BK + k_local + 2] = __float2bfloat16(vals_lo.z);
          w_tile[n_local * TC_BK + k_local + 3] = __float2bfloat16(vals_lo.w);
          w_tile[n_local * TC_BK + k_local + 4] = __float2bfloat16(vals_hi.x);
          w_tile[n_local * TC_BK + k_local + 5] = __float2bfloat16(vals_hi.y);
          w_tile[n_local * TC_BK + k_local + 6] = __float2bfloat16(vals_hi.z);
          w_tile[n_local * TC_BK + k_local + 7] = __float2bfloat16(vals_hi.w);
        }
      } else {
        #pragma unroll
        for (int j = 0; j < vals_per_pack; j++) {
          if (k_local + j < TC_BK) {
            w_tile[n_local * TC_BK + k_local + j] = __float2bfloat16(0.0f);
          }
        }
      }
    }

    __syncthreads();

    // Compute with tensor cores
    // Each warp computes a 32x64 portion using 2x4 16x16 tiles
    #pragma unroll
    for (int k = 0; k < TC_BK / 16; k++) {
      // Load A fragments (2 tiles along M)
      Tile16x16_bf16 A[2];
      const int a_row = warp_offset_m + (laneid % 16);
      const int a_col = k * 16 + (laneid / 16) * 8;
      const uint32_t x_base = __cvta_generic_to_shared(x_tile);

      A[0].load(x_base + sizeof(__nv_bfloat16) * (a_row * TC_BK + a_col));
      A[1].load(x_base + sizeof(__nv_bfloat16) * ((a_row + 16) * TC_BK + a_col));

      // Load B fragments (4 tiles along N)
      Tile16x16_bf16 B[4];
      const int b_col = k * 16 + (laneid / 16) * 8;
      const uint32_t w_base = __cvta_generic_to_shared(w_tile);

      #pragma unroll
      for (int bn = 0; bn < 4; bn++) {
        const int b_row = warp_offset_n + bn * 16 + (laneid % 16);
        B[bn].load(w_base + sizeof(__nv_bfloat16) * (b_row * TC_BK + b_col));
      }

      // 2x4 MMA operations
      #pragma unroll
      for (int bn = 0; bn < 4; bn++) {
        mma_bf16(C[bn * 2 + 0], A[0], B[bn]);
        mma_bf16(C[bn * 2 + 1], A[1], B[bn]);
      }
    }

    __syncthreads();
  }

  // ==========================================================================
  // Write results to global memory (32x64 per warp = 2x4 16x16 tiles)
  // ==========================================================================
  const int out_row_base = bm + warp_offset_m;
  const int out_col_base = bn + warp_offset_n;

  // Check if we need bounds checking
  const bool need_bounds = (out_row_base + 32 > M) || (out_col_base + 64 > N);

  if (!need_bounds) {
    // Fast path: full tile write (2 rows x 4 cols of 16x16 tiles)
    #pragma unroll
    for (int bn = 0; bn < 4; bn++) {
      C[bn * 2 + 0].store_global(out + out_row_base * N + (out_col_base + bn * 16), N);
      C[bn * 2 + 1].store_global(out + (out_row_base + 16) * N + (out_col_base + bn * 16), N);
    }
  } else {
    // Slow path: bounds-checked write
    #pragma unroll
    for (int bn = 0; bn < 4; bn++) {
      const int col_offset = out_col_base + bn * 16;
      const int max_cols = min(16, N - col_offset);
      if (max_cols <= 0) continue;

      const int max_rows_0 = min(16, M - out_row_base);
      const int max_rows_1 = min(16, M - out_row_base - 16);

      if (max_rows_0 > 0)
        C[bn * 2 + 0].store_global_safe(out + out_row_base * N + col_offset, N, max_rows_0, max_cols);
      if (max_rows_1 > 0)
        C[bn * 2 + 1].store_global_safe(out + (out_row_base + 16) * N + col_offset, N, max_rows_1, max_cols);
    }
  }
}

// =============================================================================
// Simple kernel for small M (1-8 rows) - one output per thread
// =============================================================================
template <typename T, int bits, int group_size, bool use_mx_scale>
__global__ void fp_qmm_t_small_m_kernel(
    const T* __restrict__ x,
    const uint32_t* __restrict__ w,
    const uint8_t* __restrict__ scales,
    T* __restrict__ out,
    int M, int N, int K) {

  constexpr int pack_factor = 32 / bits;
  constexpr int vals_per_pack = (bits == 8) ? 4 : 8;

  const int row = blockIdx.y;
  const int col_start = blockIdx.x * blockDim.x + threadIdx.x;
  const int col_stride = gridDim.x * blockDim.x;

  if (row >= M) return;

  const int K_packed = K / pack_factor;
  const int K_groups = K / group_size;

  for (int col = col_start; col < N; col += col_stride) {
    float sum = 0.0f;

    for (int k = 0; k < K; k += vals_per_pack) {
      int pack_idx = k / pack_factor;
      int scale_idx = k / group_size;

      uint32_t packed = w[col * K_packed + pack_idx];
      float scale = load_scale<use_mx_scale>(scales + col * K_groups, scale_idx);

      if constexpr (bits == 8) {
        float4 wv = dequant_fp8_scaled(packed, scale);
        sum += float(x[row * K + k]) * wv.x;
        sum += float(x[row * K + k + 1]) * wv.y;
        sum += float(x[row * K + k + 2]) * wv.z;
        sum += float(x[row * K + k + 3]) * wv.w;
      } else {
        uint16_t lo = packed & 0xFFFF;
        uint16_t hi = (packed >> 16) & 0xFFFF;
        float4 wv_lo = dequant_fp4_scaled(lo, scale);
        float4 wv_hi = dequant_fp4_scaled(hi, scale);

        sum += float(x[row * K + k]) * wv_lo.x;
        sum += float(x[row * K + k + 1]) * wv_lo.y;
        sum += float(x[row * K + k + 2]) * wv_lo.z;
        sum += float(x[row * K + k + 3]) * wv_lo.w;
        sum += float(x[row * K + k + 4]) * wv_hi.x;
        sum += float(x[row * K + k + 5]) * wv_hi.y;
        sum += float(x[row * K + k + 6]) * wv_hi.z;
        sum += float(x[row * K + k + 7]) * wv_hi.w;
      }
    }

    out[row * N + col] = T(sum);
  }
}

// =============================================================================
// Public interface
// =============================================================================
void fp_qmm(
    const array& w,
    const array& scales,
    const array& x,
    array& out,
    bool transpose,
    int bits,
    int group_size,
    int M,
    int N,
    int K,
    CommandEncoder& encoder) {

  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  encoder.set_input_array(x);
  encoder.set_output_array(out);

  if (!transpose) {
    throw std::runtime_error("[fp_qmm] Non-transposed weights not yet implemented");
  }

  dispatch_float_types(out.dtype(), "fp_qmm", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {

      auto launch_small_kernel = [&](auto bits_tag, auto group_tag, auto mx_tag) {
        constexpr int BITS = decltype(bits_tag)::value;
        constexpr int GROUP = decltype(group_tag)::value;
        constexpr bool USE_MX = decltype(mx_tag)::value;

        dim3 block(256);
        dim3 grid((N + 255) / 256, M);

        encoder.add_kernel_node(
            fp_qmm_t_small_m_kernel<T, BITS, GROUP, USE_MX>,
            grid, block, 0,
            gpu_ptr<T>(x),
            gpu_ptr<uint32_t>(w),
            gpu_ptr<uint8_t>(scales),
            gpu_ptr<T>(out),
            M, N, K);
      };

      auto launch_tc_kernel = [&](auto bits_tag, auto group_tag, auto mx_tag) {
        constexpr int BITS = decltype(bits_tag)::value;
        constexpr int GROUP = decltype(group_tag)::value;
        constexpr bool USE_MX = decltype(mx_tag)::value;

        dim3 block(TC_BLOCK_SIZE);
        dim3 grid((N + TC_BN - 1) / TC_BN, (M + TC_BM - 1) / TC_BM);

        // Shared memory: x_tile[TC_BM, TC_BK] + w_tile[TC_BN, TC_BK]
        size_t shmem_size = (TC_BM * TC_BK + TC_BN * TC_BK) * sizeof(__nv_bfloat16);

        encoder.add_kernel_node(
            fp_qmm_tensor_core_kernel<T, BITS, GROUP, USE_MX>,
            grid, block, shmem_size,
            gpu_ptr<T>(x),
            gpu_ptr<uint32_t>(w),
            gpu_ptr<uint8_t>(scales),
            gpu_ptr<T>(out),
            M, N, K);
      };

      // Choose kernel based on M size
      if (M <= 8) {
        // Small M: use simple per-element kernel
        if (bits == 8 && group_size == 32) {
          launch_small_kernel(std::integral_constant<int, 8>{},
                              std::integral_constant<int, 32>{},
                              std::true_type{});
        } else if (bits == 4 && group_size == 32) {
          launch_small_kernel(std::integral_constant<int, 4>{},
                              std::integral_constant<int, 32>{},
                              std::true_type{});
        } else if (bits == 4 && group_size == 16) {
          launch_small_kernel(std::integral_constant<int, 4>{},
                              std::integral_constant<int, 16>{},
                              std::false_type{});
        }
      } else {
        // Larger M: use tensor core kernel
        if (bits == 8 && group_size == 32) {
          launch_tc_kernel(std::integral_constant<int, 8>{},
                           std::integral_constant<int, 32>{},
                           std::true_type{});
        } else if (bits == 4 && group_size == 32) {
          launch_tc_kernel(std::integral_constant<int, 4>{},
                           std::integral_constant<int, 32>{},
                           std::true_type{});
        } else if (bits == 4 && group_size == 16) {
          launch_tc_kernel(std::integral_constant<int, 4>{},
                           std::integral_constant<int, 16>{},
                           std::false_type{});
        }
      }
    }
  });
}

} // namespace mlx::core::cu
