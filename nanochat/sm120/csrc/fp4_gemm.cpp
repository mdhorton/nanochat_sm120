// A drop-in for the block-scaled NVFP4 `torch._scaled_mm` in nanochat/sm120/nvfp4.py::fp4_mm,
// driving cuBLASLt directly so the launch, the algorithm and the epilogue are ours.
//
// **The algorithm choice this was built for turned out not to exist**: cuBLASLt has one
// block-scaled fp4 kernel for these shapes (algoId 70, tile 128x128, stages 256xAUTO), and
// picking the best candidate for every GEMM in the model is 1.001x (dev/nvfp4-quartet.md, *The
// GEMM headroom is not an algorithm choice*). What this file is for is the seam: queue items A2
// (per-tensor scale as a device-pointer alpha) and A3 (beta=1 wgrad accumulate into fp32
// main_grad) are epilogue changes `_scaled_mm` cannot express at all, and they need a launcher
// we own. One plan per shape, autotuned once at first use, then reused for every launch.
//
// Three plan kinds, and which one a shape gets is fixed when it is built:
//
//   plain        D = A.B, bf16 out, alpha=1 on the host          (run)
//   device alpha D = alpha.A.B, bf16 out, alpha a device pointer (run_alpha, queue A2)
//   accumulate   D = alpha.A.B + C with C = D = a caller-owned fp32 buffer and beta=1, so
//                gradient accumulation happens in the epilogue   (run_accum, queue A3)
//
// An accumulating plan always carries the device alpha: beta=1 adds the *scaled* product, so a
// per-tensor scale applied afterwards would rescale the running sum too.
//
// The descriptor recipe is not guessed -- it is what `_scaled_mm` itself issues, read off a
// CUBLASLT_LOG_LEVEL=2 trace of the real call:
//
//   Adesc=[type=R_4F_E2M1 rows=1280 cols=1280 ld=1280] Bdesc=[type=R_4F_E2M1 rows=1280
//   cols=16384 ld=1280] Ddesc=[type=R_16BF rows=1280 cols=16384 ld=1280]
//   computeType=COMPUTE_32F scaleType=R_32F transa=OP_T aScaleMode=VEC16_UE4M3
//   bScaleMode=VEC16_UE4M3 ... workSpaceSizeInBytes=1048576
//
// Two things in that trace are load-bearing:
//
// Layout. Both operands are k-major (TN), and to return a *contiguous row-major* [m,n] output
// the operands are swapped and D is computed transposed:
//     D'[n,m] col-major = op_T(b)[n,k] . op_N(a)[k,m] = (a.b)^T
// and a row-major [m,n] torch tensor is bit-identical to a col-major [n,m] one with ld=n. The
// scales swap with their operands, so cuBLASLt's "A scale" is b's. Getting this wrong is silent
// wherever m == n -- at d20 that is every attention projection.
//
// Workspace. torch queries the heuristic with 1 MiB, which is under what the split-K variants
// need (6.5 MB at the shapes that have them), so at 32 MiB the list gains candidates rather than
// merely being re-ordered. Measured: it buys one shape of nine, 1.02x on 1280x1280x8192.
//
// Another trace-only fact, and the one that costs an afternoon: with a VEC16_UE4M3 scale mode
// set, `cublasLtMatmulAlgoGetHeuristic` fails with NOT_SUPPORTED unless the scale *pointers* are
// already on the descriptor -- "requires a custom pointer for A scale". The query needs them
// even though it never dereferences them.
//
// ld is in *elements* for R_4F_E2M1 (k elements = k/2 bytes), which is why every operand here is
// taken as a packed uint8 [rows, cols/2] tensor -- the form the Quartet kernels emit -- and the
// layout is built from the unpacked k.
//
// Build (JIT) -- see nanochat/sm120/fp4_gemm.py. **A .cpp, not a .cu**: there is no device code
// here, only cuBLASLt host calls, and nvcc 12.8 hits an internal compiler error on GCC 13's
// <bits/stl_construct.h> at -std=c++20 (which this repo requires everywhere, because torch
// 2.9.1's ATen/core/List_inl.h does not compile under GCC 13 at C++17). The host compiler builds
// it in seconds and needs no gencode flag.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#define LT_CHECK(expr)                                                              \
    do {                                                                            \
        cublasStatus_t _s = (expr);                                                 \
        TORCH_CHECK(_s == CUBLAS_STATUS_SUCCESS, #expr " failed: ", int(_s));       \
    } while (0)

namespace {

// Config attributes are not all the same width -- CLUSTER_SHAPE_ID and INNER_SHAPE_ID are
// uint16_t while the rest are int32_t, and cuBLASLt rejects a read of the wrong size rather than
// truncating. Asking for 4 bytes everywhere reports every cluster shape as "-1", which is what
// the first version of this did and what made split-K look like the only axis in the candidate
// list. Returns -1 only when the attribute genuinely is not set.
int get_cfg(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr) {
    size_t written = 0;
    int32_t v32 = -1;
    if (cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &v32, sizeof(v32), &written)
        == CUBLAS_STATUS_SUCCESS) {
        return v32;
    }
    uint16_t v16 = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &v16, sizeof(v16), &written)
        == CUBLAS_STATUS_SUCCESS) {
        return int(v16);
    }
    return -1;
}

// One workspace shared by every plan: calls are sequential on one stream, and a per-plan
// allocation would cost 32 MB times the number of pinned shapes for no benefit. Plans may ask
// the heuristic for less than this (torch asks for 1 MiB, and what that costs is a measurement,
// not a guess -- see `probe`), but never for more.
constexpr int64_t kWorkspaceBytes = 32 * 1024 * 1024;

torch::Tensor& workspace(const torch::TensorOptions& opts) {
    static torch::Tensor ws;
    if (!ws.defined()) ws = torch::empty({kWorkspaceBytes}, opts.dtype(torch::kUInt8));
    return ws;
}

// [1.0, 0.0] on device. CUBLASLT_POINTER_MODE_DEVICE puts *both* alpha and beta in device
// memory, so a plan with a device alpha still needs a device beta -- and autotuning one before
// the caller has an alpha needs a device 1.0 to point at.
torch::Tensor& device_scalars(const torch::TensorOptions& opts) {
    static torch::Tensor s;
    if (!s.defined()) {
        s = torch::empty({2}, opts.dtype(torch::kFloat32));
        s[0] = 1.0f;
        s[1] = 0.0f;
    }
    return s;
}

}  // namespace

// One pinned NVFP4 GEMM: fixed shape, fixed algorithm. `a` is [m, k/2] packed uint8 row-major,
// `b` is [n, k/2] packed uint8 row-major, and the product is a.b^T as [m, n] bf16.
class Fp4Gemm {
public:
    Fp4Gemm(int64_t m, int64_t n, int64_t k, int64_t ws_bytes = kWorkspaceBytes,
            bool device_alpha = false, bool accum = false)
        : m_(m), n_(n), k_(k), ws_bytes_(ws_bytes), device_alpha_(device_alpha), accum_(accum) {
        TORCH_CHECK(k % 16 == 0, "k must be a multiple of the 16-element block scale");
        TORCH_CHECK(ws_bytes > 0 && ws_bytes <= kWorkspaceBytes,
                    "workspace must fit the shared buffer (", kWorkspaceBytes, " bytes)");
        TORCH_CHECK(!accum_ || device_alpha_,
                    "an accumulating plan needs the device alpha: beta=1 adds the scaled "
                    "product, so a per-tensor scale applied afterwards would rescale the sum");
        LT_CHECK(cublasLtCreate(&lt_));
        LT_CHECK(cublasLtMatmulDescCreate(&desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t op_a = CUBLAS_OP_T, op_b = CUBLAS_OP_N;
        LT_CHECK(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_TRANSA,
                                                &op_a, sizeof(op_a)));
        LT_CHECK(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_TRANSB,
                                                &op_b, sizeof(op_b)));
        // Per-16 E4M3 block scales on both operands. Set on the descriptor, so the heuristic is
        // queried for the block-scaled kernel family and not for a plain fp4 one.
        const int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        LT_CHECK(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                &mode, sizeof(mode)));
        LT_CHECK(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                &mode, sizeof(mode)));
        // The per-tensor scale as the GEMM's own alpha (queue A2). It has to live on the
        // descriptor before the heuristic query, like the block-scale pointers: pointer mode is
        // a capability the candidate list is filtered on, not a launch-time argument.
        if (device_alpha_) {
            const int32_t pm = CUBLASLT_POINTER_MODE_DEVICE;
            LT_CHECK(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                                    &pm, sizeof(pm)));
        }
        // Operands swapped (see file header): Adesc describes b, Bdesc describes a. An
        // accumulating plan reads and writes the caller's fp32 accumulator through this one
        // layout -- C and D are the same buffer, so they are the same descriptor.
        LT_CHECK(cublasLtMatrixLayoutCreate(&Adesc_, CUDA_R_4F_E2M1, k, n, k));
        LT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc_, CUDA_R_4F_E2M1, k, m, k));
        LT_CHECK(cublasLtMatrixLayoutCreate(&Ddesc_, accum_ ? CUDA_R_32F : CUDA_R_16BF, n, m, n));
    }

    Fp4Gemm(const Fp4Gemm&) = delete;
    Fp4Gemm& operator=(const Fp4Gemm&) = delete;

    ~Fp4Gemm() {
        if (Ddesc_) cublasLtMatrixLayoutDestroy(Ddesc_);
        if (Bdesc_) cublasLtMatrixLayoutDestroy(Bdesc_);
        if (Adesc_) cublasLtMatrixLayoutDestroy(Adesc_);
        if (desc_) cublasLtMatmulDescDestroy(desc_);
        if (lt_) cublasLtDestroy(lt_);
    }

    // Enumerate every candidate, screen, then score the finalists fairly. Returns the winner's
    // config plus the heuristic's own pick measured in the same pass, so the caller can log a
    // ratio that means something. Protocol lifted from the fp8 pinned GEMM: a single timed pass
    // in heuristic order times candidate 0 -- cuBLAS's own pick -- on a card still at its cold
    // boost and every later candidate slower, which biases both the ratio and the selection.
    std::map<std::string, double> autotune(torch::Tensor a, torch::Tensor b,
                                           torch::Tensor a_scale, torch::Tensor b_scale,
                                           int64_t max_algos, int64_t iters,
                                           int64_t finalists, int64_t rounds) {
        check_operands(a, b, a_scale, b_scale);
        // Before the heuristic, not just before the launch: with a VEC16_UE4M3 scale mode set,
        // cublasLtMatmulAlgoGetHeuristic returns NOT_SUPPORTED unless the scale pointers are
        // already on the descriptor ("requires a custom pointer for A scale").
        set_scale_pointers(a_scale, b_scale);
        auto results = heuristic(max_algos);
        const int returned = int(results.size());

        auto out = make_out(a);

        // Phase 1, screen. Time every candidate cheaply, in heuristic order. These numbers are
        // biased cool-to-hot and are used only to rank -- the pass exists mostly to be the soak
        // that takes the card off its cold boost, on the shape being scored.
        const int64_t screen_iters = std::max<int64_t>(3, iters / 4);
        std::vector<std::pair<double, int>> screened;
        std::map<int, double> screen_us;
        int heuristic_i = -1;
        for (int i = 0; i < returned; ++i) {
            if (results[i].state != CUBLAS_STATUS_SUCCESS) continue;
            if (launch(results[i].algo, a, b, a_scale, b_scale, out) != CUBLAS_STATUS_SUCCESS)
                continue;
            if (heuristic_i < 0) heuristic_i = i;  // cuBLAS's pick = first one that runs
            double us = time_algo(results[i].algo, a, b, a_scale, b_scale, out, screen_iters);
            screened.emplace_back(us, i);
            screen_us[i] = us;
        }
        TORCH_CHECK(!screened.empty(), "no cuBLASLt algorithm ran successfully for this fp4 shape");

        // Phase 2, score. Keep the K fastest and always the heuristic's pick, so the ratio is
        // measured against it in the same pass as the winner. Alternate the order each round so
        // residual drift lands on every finalist equally; take the median.
        std::sort(screened.begin(), screened.end());
        std::vector<int> pool;
        for (size_t j = 0; j < screened.size() && int64_t(pool.size()) < finalists; ++j)
            pool.push_back(screened[j].second);
        if (std::find(pool.begin(), pool.end(), heuristic_i) == pool.end())
            pool.push_back(heuristic_i);

        std::map<int, std::vector<double>> times;
        for (int64_t r = 0; r < rounds; ++r) {
            std::vector<int> order = pool;
            if (r % 2) std::reverse(order.begin(), order.end());
            for (int i : order)
                times[i].push_back(time_algo(results[i].algo, a, b, a_scale, b_scale, out, iters));
        }
        auto median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            size_t h = v.size() / 2;
            return v.size() % 2 ? v[h] : 0.5 * (v[h - 1] + v[h]);
        };

        int best_i = pool.front();
        double best_us = median(times[best_i]);
        for (int i : pool) {
            double us = median(times[i]);
            if (us < best_us) { best_us = us; best_i = i; }
        }

        algo_ = results[best_i].algo;
        pinned_ = true;
        std::map<std::string, double> info;
        info["us"] = best_us;
        info["heuristic_us"] = median(times[heuristic_i]);
        info["candidates"] = double(returned);
        info["ran"] = double(screened.size());
        info["finalists"] = double(pool.size());
        info["rounds"] = double(rounds);
        info["tile"] = get_cfg(algo_, CUBLASLT_ALGO_CONFIG_TILE_ID);
        info["splitk"] = get_cfg(algo_, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
        info["reduction"] = get_cfg(algo_, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
        info["stages"] = get_cfg(algo_, CUBLASLT_ALGO_CONFIG_STAGES_ID);
        info["algo_id"] = get_cfg(algo_, CUBLASLT_ALGO_CONFIG_ID);
        info["cluster"] = get_cfg(algo_, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID);
        info["waves"] = results[best_i].wavesCount;
        // What the heuristic's own pick looks like, for the log: a plan that only re-ranks the
        // same tile is a different claim from one that reaches a tile torch is never offered.
        info["heuristic_algo_id"] = get_cfg(results[heuristic_i].algo, CUBLASLT_ALGO_CONFIG_ID);
        info["heuristic_tile"] = get_cfg(results[heuristic_i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
        return info;
    }

    // Every candidate cuBLASLt offers for this shape and workspace budget, timed in heuristic
    // order. Diagnostic, not used in training: the question it answers is whether there is an
    // algorithm *choice* here at all, or one kernel with cosmetic variations.
    std::vector<std::map<std::string, double>> probe(torch::Tensor a, torch::Tensor b,
                                                     torch::Tensor a_scale, torch::Tensor b_scale,
                                                     int64_t max_algos, int64_t iters) {
        check_operands(a, b, a_scale, b_scale);
        set_scale_pointers(a_scale, b_scale);
        auto results = heuristic(max_algos);
        auto out = make_out(a);
        std::vector<std::map<std::string, double>> rows;
        for (size_t i = 0; i < results.size(); ++i) {
            std::map<std::string, double> row;
            row["index"] = double(i);
            row["algo_id"] = get_cfg(results[i].algo, CUBLASLT_ALGO_CONFIG_ID);
            row["tile"] = get_cfg(results[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
            row["stages"] = get_cfg(results[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
            row["splitk"] = get_cfg(results[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
            row["reduction"] = get_cfg(results[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
            row["cluster"] = get_cfg(results[i].algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID);
            row["inner"] = get_cfg(results[i].algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID);
            row["ws_required"] = double(results[i].workspaceSize);
            row["waves"] = results[i].wavesCount;
            row["us"] = (results[i].state == CUBLAS_STATUS_SUCCESS &&
                         launch(results[i].algo, a, b, a_scale, b_scale, out) == CUBLAS_STATUS_SUCCESS)
                        ? time_algo(results[i].algo, a, b, a_scale, b_scale, out, iters)
                        : -1.0;
            rows.push_back(row);
        }
        return rows;
    }

    torch::Tensor run(torch::Tensor a, torch::Tensor b,
                      torch::Tensor a_scale, torch::Tensor b_scale) {
        // Plan type before tuning state: calling the wrong one of the two is a programming
        // error whatever the plan's state, and saying so beats "before autotune()".
        TORCH_CHECK(!device_alpha_, "device-alpha plan: use run_alpha()");
        TORCH_CHECK(pinned_, "Fp4Gemm.run() before autotune()");
        check_operands(a, b, a_scale, b_scale);
        auto out = make_out(a);
        auto st = launch(algo_, a, b, a_scale, b_scale, out);
        TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "pinned fp4 cublasLtMatmul failed: ", int(st));
        return out;
    }

    // out = alpha * a.b^T, with the per-tensor scale applied inside the epilogue rather than by
    // a separate pass over [m, n]. `alpha` is a 0-d fp32 CUDA tensor -- read by the kernel, so
    // it must not be a host scalar and must outlive the launch.
    torch::Tensor run_alpha(torch::Tensor a, torch::Tensor b, torch::Tensor a_scale,
                            torch::Tensor b_scale, torch::Tensor alpha) {
        TORCH_CHECK(device_alpha_, "host-alpha plan: use run()");
        TORCH_CHECK(!accum_, "accumulating plan: use run_accum()");
        TORCH_CHECK(pinned_, "Fp4Gemm.run_alpha() before autotune()");
        check_alpha(alpha);
        check_operands(a, b, a_scale, b_scale);
        auto out = make_out(a);
        auto st = launch(algo_, a, b, a_scale, b_scale, out, alpha);
        TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "pinned fp4 alpha cublasLtMatmul failed: ", int(st));
        return out;
    }

    // out += alpha * a.b^T, in fp32, inside the epilogue (queue A3): `out` is the caller's
    // gradient accumulator and is both C and D, so the micro-step's contribution joins the
    // running sum without a separate tensor for a separate kernel to add. Returns nothing --
    // the result is the mutation, which is what the custom op around it declares.
    void run_accum(torch::Tensor a, torch::Tensor b, torch::Tensor a_scale,
                   torch::Tensor b_scale, torch::Tensor alpha, torch::Tensor out) {
        TORCH_CHECK(accum_, "non-accumulating plan: use run() or run_alpha()");
        TORCH_CHECK(pinned_, "Fp4Gemm.run_accum() before autotune()");
        check_alpha(alpha);
        check_operands(a, b, a_scale, b_scale);
        TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32 && out.dim() == 2 &&
                    out.size(0) == m_ && out.size(1) == n_ && out.is_contiguous(),
                    "the accumulator must be a contiguous [m, n] fp32 CUDA tensor");
        auto st = launch(algo_, a, b, a_scale, b_scale, out, alpha);
        TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "pinned fp4 accum cublasLtMatmul failed: ", int(st));
    }

private:
    // An accumulating plan reads C before it writes D, so its scratch has to be zeroed rather
    // than empty: uninitialized fp32 can hold NaN, which beta=1 would fold into every timed
    // iteration and into the verification that follows.
    torch::Tensor make_out(const torch::Tensor& a) const {
        return accum_ ? torch::zeros({m_, n_}, a.options().dtype(torch::kFloat32))
                      : torch::empty({m_, n_}, a.options().dtype(torch::kBFloat16));
    }

    void check_alpha(const torch::Tensor& alpha) const {
        TORCH_CHECK(alpha.is_cuda() && alpha.scalar_type() == torch::kFloat32 &&
                    alpha.numel() == 1 && alpha.is_contiguous(),
                    "alpha must be a contiguous single-element fp32 CUDA tensor");
    }

    void check_operands(const torch::Tensor& a, const torch::Tensor& b,
                        const torch::Tensor& a_scale, const torch::Tensor& b_scale) const {
        TORCH_CHECK(a.is_cuda() && b.is_cuda(), "operands must be CUDA tensors");
        TORCH_CHECK(a.scalar_type() == torch::kUInt8 && b.scalar_type() == torch::kUInt8,
                    "operands must be packed uint8 (two fp4 codes per byte)");
        TORCH_CHECK(a.dim() == 2 && a.size(0) == m_ && a.size(1) == k_ / 2,
                    "a must be [m, k/2]");
        TORCH_CHECK(b.dim() == 2 && b.size(0) == n_ && b.size(1) == k_ / 2,
                    "b must be [n, k/2]");
        TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "operands must be row-major contiguous");
        // Block scales: one E4M3 byte per 16 elements, in the cutlass 128x4 swizzle the Quartet
        // kernels already emit. The swizzle is not checkable from here -- a row-major scale
        // block returns confident garbage rather than an error, which is what the numerical
        // verification in fp4_gemm.py exists to catch.
        TORCH_CHECK(a_scale.is_cuda() && b_scale.is_cuda(), "scales must be CUDA tensors");
        TORCH_CHECK(a_scale.numel() == m_ * k_ / 16 && b_scale.numel() == n_ * k_ / 16,
                    "block scales must hold one byte per 16 elements");
        TORCH_CHECK(a_scale.is_contiguous() && b_scale.is_contiguous(),
                    "block scales must be contiguous");
        TORCH_CHECK(a_scale.element_size() == 1 && b_scale.element_size() == 1,
                    "block scales must be an 8-bit type (E4M3)");
    }

    // The candidate list for this shape at this plan's workspace budget. The budget is part of
    // the question: torch queries with 1 MiB, which is not enough for the split-K variants.
    std::vector<cublasLtMatmulHeuristicResult_t> heuristic(int64_t max_algos) {
        cublasLtMatmulPreference_t pref = nullptr;
        size_t ws_bytes = size_t(ws_bytes_);
        LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
        std::vector<cublasLtMatmulHeuristicResult_t> results(max_algos);
        int returned = 0;
        LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt_, desc_, Adesc_, Bdesc_, Ddesc_, Ddesc_, pref,
                                                int(max_algos), results.data(), &returned));
        cublasLtMatmulPreferenceDestroy(pref);
        TORCH_CHECK(returned > 0, "cuBLASLt returned no algorithms for this fp4 shape");
        results.resize(returned);
        return results;
    }

    // Scales follow their operands through the swap: cuBLASLt's A is b.
    void set_scale_pointers(const torch::Tensor& a_scale, const torch::Tensor& b_scale) {
        const void* a_sp = b_scale.data_ptr();
        const void* b_sp = a_scale.data_ptr();
        LT_CHECK(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                &a_sp, sizeof(a_sp)));
        LT_CHECK(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                &b_sp, sizeof(b_sp)));
    }

    // `alpha` is only consulted by a device-alpha plan; a host-alpha plan always runs alpha=1
    // and leaves the per-tensor scale to the caller. beta is 1 for an accumulating plan and 0
    // otherwise, and in device pointer mode it has to come from device memory too -- hence the
    // two-element [1.0, 0.0] constant rather than a device 0.0 alone.
    cublasStatus_t launch(const cublasLtMatmulAlgo_t& algo, const torch::Tensor& a,
                          const torch::Tensor& b, const torch::Tensor& a_scale,
                          const torch::Tensor& b_scale, torch::Tensor& out,
                          const torch::Tensor& alpha = {}) {
        const float host_alpha = 1.0f, host_beta = accum_ ? 1.0f : 0.0f;
        const void* alpha_p = &host_alpha;
        const void* beta_p = &host_beta;
        if (device_alpha_) {
            auto& s = device_scalars(out.options());
            alpha_p = alpha.defined() ? alpha.data_ptr() : s.data_ptr();
            beta_p = static_cast<const float*>(s.data_ptr()) + (accum_ ? 0 : 1);
        }
        set_scale_pointers(a_scale, b_scale);
        auto& ws = workspace(out.options());
        return cublasLtMatmul(lt_, desc_, alpha_p, b.data_ptr(), Adesc_, a.data_ptr(), Bdesc_,
                              beta_p, out.data_ptr(), Ddesc_, out.data_ptr(), Ddesc_, &algo,
                              ws.data_ptr(), size_t(ws_bytes_), at::cuda::getCurrentCUDAStream());
    }

    double time_algo(const cublasLtMatmulAlgo_t& algo, const torch::Tensor& a,
                     const torch::Tensor& b, const torch::Tensor& a_scale,
                     const torch::Tensor& b_scale, torch::Tensor& out, int64_t iters) {
        auto stream = at::cuda::getCurrentCUDAStream();
        for (int w = 0; w < 3; ++w) launch(algo, a, b, a_scale, b_scale, out);
        cudaEvent_t t0, t1;
        C10_CUDA_CHECK(cudaEventCreate(&t0));
        C10_CUDA_CHECK(cudaEventCreate(&t1));
        C10_CUDA_CHECK(cudaEventRecord(t0, stream));
        for (int64_t it = 0; it < iters; ++it) launch(algo, a, b, a_scale, b_scale, out);
        C10_CUDA_CHECK(cudaEventRecord(t1, stream));
        C10_CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0.f;
        C10_CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        C10_CUDA_CHECK(cudaEventDestroy(t0));
        C10_CUDA_CHECK(cudaEventDestroy(t1));
        return double(ms) * 1000.0 / double(iters);
    }

    int64_t m_, n_, k_, ws_bytes_;
    bool device_alpha_ = false;
    bool accum_ = false;
    bool pinned_ = false;
    cublasLtHandle_t lt_ = nullptr;
    cublasLtMatmulDesc_t desc_ = nullptr;
    cublasLtMatrixLayout_t Adesc_ = nullptr, Bdesc_ = nullptr, Ddesc_ = nullptr;
    cublasLtMatmulAlgo_t algo_{};
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cublaslt_version", []() { return int64_t(cublasLtGetVersion()); },
          "Runtime cuBLASLt version -- which library got loaded, not which header built this.");
    m.attr("WORKSPACE_BYTES") = int64_t(kWorkspaceBytes);
    pybind11::class_<Fp4Gemm>(m, "Fp4Gemm")
        .def(pybind11::init<int64_t, int64_t, int64_t, int64_t, bool, bool>(),
             pybind11::arg("m"), pybind11::arg("n"), pybind11::arg("k"),
             pybind11::arg("ws_bytes") = kWorkspaceBytes,
             pybind11::arg("device_alpha") = false,
             pybind11::arg("accum") = false)
        .def("autotune", &Fp4Gemm::autotune)
        .def("probe", &Fp4Gemm::probe)
        .def("run", &Fp4Gemm::run)
        .def("run_alpha", &Fp4Gemm::run_alpha)
        .def("run_accum", &Fp4Gemm::run_accum);
}
