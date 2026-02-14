// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/backend/cuda/quantized/qmv.h"
#include "mlx/backend/cuda/quantized/quantized_utils.h"
#include "mlx/backend/common/matmul.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("QuantizedMatmul::eval_gpu");
  auto& s = stream();
  auto& d = cu::device(s.device);
  auto& enc = d.get_command_encoder(s);

  out.set_data(cu::malloc_async(out.nbytes(), enc));

  // Make sure the last two dims of x and w, s, b are contiguous. This should
  // be relaxed for x.
  array x = ensure_row_contiguous_matrix(inputs[0], enc, s);
  array w = ensure_row_contiguous_matrix(inputs[1], enc, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], enc, s);
  std::optional<array> biases = std::nullopt;
  if (inputs.size() == 4) {
    biases = ensure_row_contiguous_matrix(inputs[3], enc, s);
  }

  bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);

  // Affine mode is not yet implemented
  if (mode_ == QuantizationMode::Affine) {
    throw std::runtime_error("QMM NYI");
  }

  // For M=1 with transpose, use the optimized fp_qmv kernel
  if (M == 1 && transpose_) {
    fp_qmv(w, scales, x, out, bits_, group_size_, M, N, K, enc);
    return;
  }

  // For M>1 or non-transposed, dequantize weights and use cuBLAS GEMM
  // Allocate temporary buffer for dequantized weights
  auto w_shape = w.shape();
  auto w_dq_shape = w_shape;
  int pack_factor = 32 / bits_;

  // Determine weight dimensions based on transpose mode
  // transpose_=true:  w is [N, K/pack_factor] packed -> [N, K] dequantized
  // transpose_=false: w is [K, N/pack_factor] packed -> [K, N] dequantized
  int w_outer = transpose_ ? N : K;
  int w_inner = transpose_ ? K : N;
  w_dq_shape.back() = w_inner;

  array w_dequant(
      cu::malloc_async(w_outer * w_inner * size_of(x.dtype()), enc),
      std::move(w_dq_shape),
      x.dtype());
  enc.add_temporary(w_dequant);

  // Dequantize weights
  fp_dequantize(w, scales, w_dequant, group_size_, bits_, enc, s);

  // Compute matrix multiplication using cuBLAS
  // transpose_=true:  x @ w_dequant.T = [M, K] @ [K, N] = [M, N] (w_dequant is [N, K])
  // transpose_=false: x @ w_dequant   = [M, K] @ [K, N] = [M, N] (w_dequant is [K, N])
  auto [batch_shape, a_batch_strides, b_batch_strides] =
      collapse_batches(x, w_dequant);
  auto batch_count = out.size() / (M * N);

  // cuBLAS uses logical dimensions: b_rows=K (inner), b_cols=N (outer)
  // The b_transposed flag tells cuBLAS how data is stored:
  //   transpose_=true:  w_dequant is [N, K] stored, so b_transposed=true, ldb=K
  //   transpose_=false: w_dequant is [K, N] stored, so b_transposed=false, ldb=N
  CublasGemm gemm(
      d,
      x.dtype(),
      /*a_transposed=*/false,
      /*a_rows=*/M,
      /*a_cols=*/K,
      /*lda=*/K,
      /*b_transposed=*/transpose_,
      /*b_rows=*/K,
      /*b_cols=*/N,
      /*ldb=*/transpose_ ? K : N,
      /*batch_count=*/batch_count,
      /*a_batch_stride=*/M * K,
      /*b_batch_stride=*/0); // weights not batched

  gemm.run(enc, out, x, w_dequant, batch_shape, a_batch_strides, b_batch_strides);
}

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("Quantize::eval_gpu");
  auto& s = stream();
  auto& d = cu::device(s.device);
  auto& enc = d.get_command_encoder(s);

  if (dequantize_) {
    auto wq = ensure_row_contiguous(inputs[0], enc, s);
    auto scales = ensure_row_contiguous(inputs[1], enc, s);
    auto& w = outputs[0];

    w.set_data(cu::malloc_async(w.nbytes(), enc));

    if (mode_ == QuantizationMode::Affine) {
      auto biases = ensure_row_contiguous(inputs[2], enc, s);
      affine_dequantize(wq, scales, biases, w, group_size_, bits_, enc, s);
    } else {
      fp_dequantize(wq, scales, w, group_size_, bits_, enc, s);
    }
  } else {
    auto w = ensure_contiguous(inputs[0], enc, s);
    auto& wq = outputs[0];
    auto& scales = outputs[1];

    wq.set_data(cu::malloc_async(wq.nbytes(), enc));
    scales.set_data(cu::malloc_async(scales.nbytes(), enc));
    if (mode_ == QuantizationMode::Affine) {
      auto& biases = outputs[2];
      biases.set_data(cu::malloc_async(biases.nbytes(), enc));
      affine_quantize(w, wq, scales, biases, group_size_, bits_, enc, s);
    } else {
      fp_quantize(w, wq, scales, group_size_, bits_, enc, s);
    }
  }
}

} // namespace mlx::core
