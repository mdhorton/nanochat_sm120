// Torch bindings for the vendored Quartet-II NVFP4 kernels.
//
// Upstream (IST-DASLab/Quartet-II) binds these through nanobind + dlpack, which drags in
// scikit-build-core, CMake and a cp312 stable-ABI wheel. The .cu/.cuh files themselves take
// raw pointers and include no binding headers at all, so they are vendored verbatim and this
// file replaces upstream's csrc/binding.cpp. Function names, argument order and the tensor
// contracts are kept identical to upstream's NB_MODULE so quant.py/rht.py stay a thin port.
//
// Stream: the kernels launch with a bare <<<>>>, i.e. the legacy default stream, which is also
// PyTorch's default stream. Calling these from a side stream or inside a CUDA graph is not
// supported -- that would need the launches themselves to take a stream argument.

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

// ---------------------------------------------------------------------------------------------
// Declarations of the vendored kernels (see csrc/*.cu).
// ---------------------------------------------------------------------------------------------

void group_transform_128_eden(__nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
                              nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h, const nv_bfloat16* x,
                              long seed, float fp4_max, float fp8_max, int M, int N, bool transpose);
extern void group_transform_128(nv_bfloat16* y, const nv_bfloat16* trans, const nv_bfloat16* x, int M, int N, bool transpose);

void transform_rht128(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, bool transpose);
void transform_rht128_ws(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, bool transpose);
void transform_rht128_tma(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, bool transpose);

void transform_rht128_eden(__nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h, const nv_bfloat16* x,
    long seed, float fp4_max, float fp8_max, int M, int N, bool transpose);

void rht128_requant(
    __nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h,
    const __nv_fp4x2_storage_t* x, const __nv_fp8_e4m3* x_scales, const float* x_global_scale,
    long seed, float fp4_max, float fp8_max, int M, int N);

void eden_fp4(__nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr, const nv_bfloat16* x_ptr, const float* amax_ptr, float scale_override, long seed, long rows, long cols);
void rtn_fp4(__nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr, const nv_bfloat16* x_ptr, const float* amax_ptr, float scale_override, long rows, long cols);
void four_six_fp4(__nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr, const nv_bfloat16* x_ptr, const float* amax_ptr, float scale_override, long rows, long cols);

void dequant_tp_had_quant(
    __nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h,
    const __nv_fp4x2_storage_t* x, const __nv_fp8_e4m3* x_scales, const float* x_global_scale,
    long seed, float fp4_max, float fp8_max, int M, int N);

// ---------------------------------------------------------------------------------------------
// Checks. Upstream's nanobind signatures encode dtype/contiguity/shape in the type; with
// at::Tensor they have to be spelled out, so these mirror the CudaArray<...> template args.
// ---------------------------------------------------------------------------------------------

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous(), #x " must be contiguous CUDA")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).scalar_type() == (dt), #x " must be " #dt ", got ", (x).scalar_type())
#define CHECK_SCALAR(x, dt) do { CHECK_CUDA(x); CHECK_DTYPE(x, dt); TORCH_CHECK((x).dim() == 0, #x " must be 0-d"); } while (0)

namespace {

// The seed arrives as a 0-d CPU int64 tensor rather than a Python int: upstream does the same,
// and it keeps the value an *input* under torch.compile instead of a constant baked at trace
// time, which would make every replay of a compiled graph reuse one stochastic-rounding seed.
long seed_value(const at::Tensor& seed) {
    TORCH_CHECK(seed.device().is_cpu(), "seed must be a CPU tensor");
    TORCH_CHECK(seed.scalar_type() == at::kLong, "seed must be int64");
    TORCH_CHECK(seed.dim() == 0, "seed must be 0-d");
    return seed.item<long>();
}

template <typename T>
T* ptr(const at::Tensor& t) {
    return reinterpret_cast<T*>(t.data_ptr());
}

template <typename T>
const T* cptr(const at::Tensor& t) {
    return reinterpret_cast<const T*>(t.data_ptr());
}

// Shared shape contract of the four fused transform+quantize entry points.
void check_eden_shapes(const at::Tensor& y, const at::Tensor& group_scales, const at::Tensor& tensor_scale,
                       const at::Tensor& scratch, const at::Tensor& max_scale, const at::Tensor& x, bool transpose) {
    CHECK_CUDA(x);
    CHECK_DTYPE(x, at::kBFloat16);
    TORCH_CHECK(x.dim() == 2, "x must be 2-d");
    TORCH_CHECK(x.size(0) % 128 == 0 && x.size(1) % 128 == 0, "x dims must be multiples of 128, got ", x.sizes());

    CHECK_CUDA(y);
    CHECK_CUDA(group_scales);
    CHECK_SCALAR(tensor_scale, at::kFloat);
    CHECK_SCALAR(max_scale, at::kUInt32);
    CHECK_CUDA(scratch);
    CHECK_DTYPE(scratch, at::kBFloat16);

    if (transpose) {
        TORCH_CHECK(y.size(0) == x.size(1) && y.size(1) == x.size(0) / 2, "y shape mismatch (transpose)");
        TORCH_CHECK(group_scales.size(0) == x.size(1) && group_scales.size(1) == x.size(0) / 16, "group_scales shape mismatch (transpose)");
    } else {
        TORCH_CHECK(y.size(0) == x.size(0) && y.size(1) == x.size(1) / 2, "y shape mismatch");
        TORCH_CHECK(group_scales.size(0) == x.size(0) && group_scales.size(1) == x.size(1) / 16, "group_scales shape mismatch");
    }
    TORCH_CHECK(scratch.numel() == x.numel() / 16, "scratch must hold x.numel()/16 bf16 values");
}

// Shared shape contract of the two fp4-in / fp4-out requantize entry points. `x` is packed
// two-per-byte, so its logical column count is 2*x.size(1).
void check_requant_shapes(const at::Tensor& y, const at::Tensor& group_scales, const at::Tensor& tensor_scale,
                          const at::Tensor& scratch, const at::Tensor& max_scale, const at::Tensor& x,
                          const at::Tensor& x_group_scales, const at::Tensor& x_tensor_scale) {
    CHECK_CUDA(x);
    CHECK_DTYPE(x, at::kByte);
    TORCH_CHECK(x.dim() == 2, "x must be 2-d");
    TORCH_CHECK(x.size(0) % 128 == 0, "x rows must be a multiple of 128, got ", x.size(0));
    TORCH_CHECK(x.size(1) % 64 == 0, "x packed cols must be a multiple of 64, got ", x.size(1));

    CHECK_CUDA(y);
    CHECK_CUDA(group_scales);
    CHECK_SCALAR(tensor_scale, at::kFloat);
    CHECK_SCALAR(max_scale, at::kUInt32);
    CHECK_CUDA(scratch);
    CHECK_DTYPE(scratch, at::kBFloat16);
    CHECK_CUDA(x_group_scales);
    CHECK_SCALAR(x_tensor_scale, at::kFloat);

    TORCH_CHECK(y.size(0) == x.size(1) * 2 && y.size(1) == x.size(0) / 2, "y shape mismatch");
    TORCH_CHECK(group_scales.size(0) == x.size(1) * 2 && group_scales.size(1) == x.size(0) / 16, "group_scales shape mismatch");
    TORCH_CHECK(x_group_scales.size(0) == x.size(0) && x_group_scales.size(1) == x.size(1) / 8, "x_group_scales shape mismatch");
    TORCH_CHECK(scratch.numel() == x.numel() / 8, "scratch must hold x.numel()/8 bf16 values");
}

// Shared shape contract of the three plain (no transform) rounding entry points.
void check_round_shapes(const at::Tensor& out, const at::Tensor& scales, const at::Tensor& global_scale,
                        const at::Tensor& inp, const at::Tensor& amax) {
    CHECK_CUDA(inp);
    CHECK_DTYPE(inp, at::kBFloat16);
    TORCH_CHECK(inp.dim() == 2 && out.dim() == 2, "inp and out must be 2-d");
    CHECK_CUDA(out);
    CHECK_CUDA(scales);
    CHECK_SCALAR(global_scale, at::kFloat);
    CHECK_SCALAR(amax, at::kFloat);

    TORCH_CHECK(out.size(0) == inp.size(0), "out rows must match inp");
    TORCH_CHECK(out.numel() * 2 == inp.numel(), "out must be half of inp (two fp4 per byte)");
    TORCH_CHECK(out.element_size() == 1, "out must be a 1-byte dtype");
    TORCH_CHECK(scales.numel() * 16 == inp.numel(), "scales must be one per 16 inputs");
    TORCH_CHECK(scales.element_size() == 1, "scales must be a 1-byte dtype");
    // The kernels write the global scale before reading amax; aliasing them silently corrupts it.
    TORCH_CHECK(global_scale.data_ptr() != amax.data_ptr(), "global_scale must not alias amax");
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Bindings.
// ---------------------------------------------------------------------------------------------

void group_transform_128_binding(at::Tensor y, at::Tensor h, at::Tensor x, bool transpose) {
    CHECK_CUDA(x); CHECK_CUDA(y); CHECK_CUDA(h);
    CHECK_DTYPE(x, at::kBFloat16); CHECK_DTYPE(y, at::kBFloat16); CHECK_DTYPE(h, at::kBFloat16);
    TORCH_CHECK(h.dim() == 2 && h.size(0) == 128 && h.size(1) == 128, "h must be 128x128");
    TORCH_CHECK(x.size(0) % 128 == 0 && x.size(1) % 128 == 0, "x dims must be multiples of 128");
    if (transpose) {
        TORCH_CHECK(x.size(0) == y.size(1) && x.size(1) == y.size(0), "y shape mismatch (transpose)");
    } else {
        TORCH_CHECK(x.size(0) == y.size(0) && x.size(1) == y.size(1), "y shape mismatch");
    }
    group_transform_128(ptr<nv_bfloat16>(y), cptr<nv_bfloat16>(h), cptr<nv_bfloat16>(x),
                        x.size(0), x.size(1), transpose);
}

void transform_rht128_binding(at::Tensor y, at::Tensor h, at::Tensor x, bool transpose) {
    CHECK_CUDA(x); CHECK_CUDA(y); CHECK_CUDA(h);
    CHECK_DTYPE(x, at::kBFloat16); CHECK_DTYPE(y, at::kBFloat16); CHECK_DTYPE(h, at::kBFloat16);
    TORCH_CHECK(h.numel() == 16 * 128, "h must be the top 16x128 slice of a 128x128 Hadamard");
    TORCH_CHECK(x.size(0) % 128 == 0 && x.size(1) % 128 == 0, "x dims must be multiples of 128");
    if (transpose) {
        TORCH_CHECK(x.size(0) == y.size(1) && x.size(1) == y.size(0), "y shape mismatch (transpose)");
    } else {
        TORCH_CHECK(x.size(0) == y.size(0) && x.size(1) == y.size(1), "y shape mismatch");
    }
    // Upstream dispatches this name to the TMA variant; _ws and plain _rht128 stay unbound.
    transform_rht128_tma(ptr<nv_bfloat16>(y), cptr<nv_bfloat16>(h), cptr<nv_bfloat16>(x),
                         x.size(0), x.size(1), transpose);
}

void group_transform_128_eden_binding(at::Tensor y, at::Tensor group_scales, at::Tensor tensor_scale,
                                      at::Tensor scratch, at::Tensor max_scale, at::Tensor h, at::Tensor x,
                                      at::Tensor seed, double fp4_max, double fp8_max, bool transpose) {
    check_eden_shapes(y, group_scales, tensor_scale, scratch, max_scale, x, transpose);
    CHECK_CUDA(h); CHECK_DTYPE(h, at::kBFloat16);
    TORCH_CHECK(h.dim() == 2 && h.size(0) == 128 && h.size(1) == 128, "h must be 128x128");
    group_transform_128_eden(
        ptr<__nv_fp4x2_storage_t>(y), ptr<__nv_fp8_e4m3>(group_scales), ptr<float>(tensor_scale),
        ptr<nv_bfloat16>(scratch), ptr<unsigned>(max_scale), cptr<nv_bfloat16>(h), cptr<nv_bfloat16>(x),
        seed_value(seed), static_cast<float>(fp4_max), static_cast<float>(fp8_max),
        x.size(0), x.size(1), transpose);
}

void rht128_eden_binding(at::Tensor y, at::Tensor group_scales, at::Tensor tensor_scale,
                         at::Tensor scratch, at::Tensor max_scale, at::Tensor h, at::Tensor x,
                         at::Tensor seed, double fp4_max, double fp8_max, bool transpose) {
    check_eden_shapes(y, group_scales, tensor_scale, scratch, max_scale, x, transpose);
    CHECK_CUDA(h); CHECK_DTYPE(h, at::kBFloat16);
    TORCH_CHECK(h.numel() == 16 * 128, "h must be the top 16x128 slice of a 128x128 Hadamard");
    transform_rht128_eden(
        ptr<__nv_fp4x2_storage_t>(y), ptr<__nv_fp8_e4m3>(group_scales), ptr<float>(tensor_scale),
        ptr<nv_bfloat16>(scratch), ptr<unsigned>(max_scale), cptr<nv_bfloat16>(h), cptr<nv_bfloat16>(x),
        seed_value(seed), static_cast<float>(fp4_max), static_cast<float>(fp8_max),
        x.size(0), x.size(1), transpose);
}

void dequant_tp_had_quant_binding(at::Tensor y, at::Tensor group_scales, at::Tensor tensor_scale,
                                  at::Tensor scratch, at::Tensor max_scale, at::Tensor h, at::Tensor x,
                                  at::Tensor x_group_scales, at::Tensor x_tensor_scale,
                                  at::Tensor seed, double fp4_max, double fp8_max) {
    check_requant_shapes(y, group_scales, tensor_scale, scratch, max_scale, x, x_group_scales, x_tensor_scale);
    CHECK_CUDA(h); CHECK_DTYPE(h, at::kBFloat16);
    TORCH_CHECK(h.dim() == 2 && h.size(0) == 128 && h.size(1) == 128, "h must be 128x128");
    dequant_tp_had_quant(
        ptr<__nv_fp4x2_storage_t>(y), ptr<__nv_fp8_e4m3>(group_scales), ptr<float>(tensor_scale),
        ptr<nv_bfloat16>(scratch), ptr<unsigned>(max_scale), cptr<nv_bfloat16>(h),
        cptr<__nv_fp4x2_storage_t>(x), cptr<__nv_fp8_e4m3>(x_group_scales), cptr<float>(x_tensor_scale),
        seed_value(seed), static_cast<float>(fp4_max), static_cast<float>(fp8_max),
        x.size(0), x.size(1) * 2);
}

void rht128_requant_binding(at::Tensor y, at::Tensor group_scales, at::Tensor tensor_scale,
                            at::Tensor scratch, at::Tensor max_scale, at::Tensor h, at::Tensor x,
                            at::Tensor x_group_scales, at::Tensor x_tensor_scale,
                            at::Tensor seed, double fp4_max, double fp8_max) {
    check_requant_shapes(y, group_scales, tensor_scale, scratch, max_scale, x, x_group_scales, x_tensor_scale);
    CHECK_CUDA(h); CHECK_DTYPE(h, at::kBFloat16);
    TORCH_CHECK(h.numel() == 16 * 128, "h must be the top 16x128 slice of a 128x128 Hadamard");
    rht128_requant(
        ptr<__nv_fp4x2_storage_t>(y), ptr<__nv_fp8_e4m3>(group_scales), ptr<float>(tensor_scale),
        ptr<nv_bfloat16>(scratch), ptr<unsigned>(max_scale), cptr<nv_bfloat16>(h),
        cptr<__nv_fp4x2_storage_t>(x), cptr<__nv_fp8_e4m3>(x_group_scales), cptr<float>(x_tensor_scale),
        seed_value(seed), static_cast<float>(fp4_max), static_cast<float>(fp8_max),
        x.size(0), x.size(1) * 2);
}

void eden_fp4_binding(at::Tensor out, at::Tensor scales, at::Tensor global_scale,
                      at::Tensor inp, at::Tensor amax, double scale_override, at::Tensor seed) {
    check_round_shapes(out, scales, global_scale, inp, amax);
    eden_fp4(ptr<__nv_fp4x4_e2m1>(out), ptr<__nv_fp8_e4m3>(scales), ptr<float>(global_scale),
             cptr<nv_bfloat16>(inp), cptr<float>(amax), static_cast<float>(scale_override),
             seed_value(seed), inp.size(0), inp.size(1));
}

void four_six_fp4_binding(at::Tensor out, at::Tensor scales, at::Tensor global_scale,
                          at::Tensor inp, at::Tensor amax, double scale_override) {
    check_round_shapes(out, scales, global_scale, inp, amax);
    four_six_fp4(ptr<__nv_fp4x4_e2m1>(out), ptr<__nv_fp8_e4m3>(scales), ptr<float>(global_scale),
                 cptr<nv_bfloat16>(inp), cptr<float>(amax), static_cast<float>(scale_override),
                 inp.size(0), inp.size(1));
}

void rtn_fp4_binding(at::Tensor out, at::Tensor scales, at::Tensor global_scale,
                     at::Tensor inp, at::Tensor amax, double scale_override) {
    check_round_shapes(out, scales, global_scale, inp, amax);
    rtn_fp4(ptr<__nv_fp4x4_e2m1>(out), ptr<__nv_fp8_e4m3>(scales), ptr<float>(global_scale),
            cptr<nv_bfloat16>(inp), cptr<float>(amax), static_cast<float>(scale_override),
            inp.size(0), inp.size(1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("group_transform_128", &group_transform_128_binding,
          py::arg("y"), py::arg("h"), py::arg("x"), py::arg("transpose") = false);
    m.def("transform_rht128", &transform_rht128_binding,
          py::arg("y"), py::arg("h"), py::arg("x"), py::arg("transpose") = false);
    m.def("group_transform_128_eden", &group_transform_128_eden_binding,
          py::arg("out"), py::arg("group_scales"), py::arg("tensor_scale"), py::arg("scratch"),
          py::arg("max_scale"), py::arg("h"), py::arg("x"), py::arg("seed"),
          py::arg("fp4_max") = 6.0, py::arg("fp8_max") = 256.0, py::arg("transpose") = false);
    m.def("rht128_eden", &rht128_eden_binding,
          py::arg("out"), py::arg("group_scales"), py::arg("tensor_scale"), py::arg("scratch"),
          py::arg("max_scale"), py::arg("h"), py::arg("x"), py::arg("seed"),
          py::arg("fp4_max") = 6.0, py::arg("fp8_max") = 256.0, py::arg("transpose") = false);
    m.def("dequant_tp_had_quant", &dequant_tp_had_quant_binding,
          py::arg("out"), py::arg("out_group_scales"), py::arg("out_tensor_scale"), py::arg("scratch"),
          py::arg("max_scale"), py::arg("h"), py::arg("x"), py::arg("x_group_scales"), py::arg("x_tensor_scale"),
          py::arg("seed"), py::arg("fp4_max") = 6.0, py::arg("fp8_max") = 256.0);
    m.def("rht128_requant", &rht128_requant_binding,
          py::arg("out"), py::arg("out_group_scales"), py::arg("out_tensor_scale"), py::arg("scratch"),
          py::arg("max_scale"), py::arg("h"), py::arg("x"), py::arg("x_group_scales"), py::arg("x_tensor_scale"),
          py::arg("seed"), py::arg("fp4_max") = 6.0, py::arg("fp8_max") = 256.0);
    m.def("eden_fp4", &eden_fp4_binding,
          py::arg("out"), py::arg("scales"), py::arg("global_scale"), py::arg("input"),
          py::arg("amax"), py::arg("scale_override"), py::arg("seed"));
    m.def("four_six_fp4", &four_six_fp4_binding,
          py::arg("out"), py::arg("scales"), py::arg("global_scale"), py::arg("input"),
          py::arg("amax"), py::arg("scale_override"));
    m.def("rtn_fp4", &rtn_fp4_binding,
          py::arg("out"), py::arg("scales"), py::arg("global_scale"), py::arg("input"),
          py::arg("amax"), py::arg("scale_override"));
}
