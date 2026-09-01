// A drop-in for torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=bf16, use_fast_accum=...)
// that runs a *chosen* cuBLASLt algorithm instead of the heuristic's first pick.
//
// Experiment 7 (dev/perf-log.md) measures cuBLAS mispicking the fp8 wgrad GEMMs by 15-42%, and
// fwd/dgrad by 5-25%. `_scaled_mm` offers no way to request a different algorithm, so this
// bypasses it: one plan per (shape, dtype pair, fast_accum), autotuned once at first use, then
// reused for every launch.
//
// fast_accum is a plan property, not a call argument: it changes which algorithm is fastest, so
// a plan tuned without it may not be the right pick with it. The forward GEMM sets it (matching
// `_Float8Matmul.forward`), dgrad and wgrad do not.
//
// accum is likewise a plan property: an accum plan computes D = A.B + C with beta=1 and
// C = D = a caller-owned fp32 buffer (run_accum), folding gradient accumulation into the wgrad
// epilogue. Autotuned with beta=1 so the timing includes the epilogue read of C.
//
// Layout. FP8 on cuBLASLt is TN-only on Ada/Hopper: both operands k-major, A transposed. To
// return a *contiguous row-major* [m,n] output (what _scaled_mm returns, and what the optimizer
// expects) the operands are swapped and D is computed transposed:
//     D'[n,m] col-major = op_T(b)[n,k] . op_N(a)[k,m] = (a.b)^T
// and a row-major [m,n] torch tensor is bit-identical to a col-major [n,m] one with ld=n.
// Getting this wrong is silent for the attention wgrad, where m == n == 768 -- the result is
// transposed but the shape still matches. The scales swap with the operands.
//
// nt: the same product with both operands read in their *natural* (non-k-major) layout, which
// sm120's cuBLASLt accepts (dev/perf-log.md, the layout probe) and Ada/Hopper would refuse.
// The ops flip (TRANSA=N, TRANSB=T) and each operand's stored shape flips with them, so
// op(A)/op(B) stay the same [n,k]/[k,m] matrices and D is unchanged. Built for the wgrad,
// where a = go as [k,m] row-major and b = x as [k,n] row-major -- the two tensors the TN form
// can only consume through transpose copies (4.6% of a training step as pure-copy kernels).
//
// Build (JIT) -- see nanochat/fp8_pinned.py. -std=c++20 is REQUIRED as elsewhere in this repo:
// torch 2.9.1's ATen/core/List_inl.h does not compile under GCC 13 at C++17.

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

int get_cfg(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr) {
    int v = -1;
    size_t written = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &v, sizeof(v), &written)
        != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    return v;
}

cudaDataType_t fp8_type(torch::ScalarType t) {
    TORCH_CHECK(t == torch::kFloat8_e4m3fn || t == torch::kFloat8_e5m2, "operand must be fp8");
    return t == torch::kFloat8_e4m3fn ? CUDA_R_8F_E4M3 : CUDA_R_8F_E5M2;
}

// One workspace shared by every plan: calls are sequential on one stream, and a per-plan
// allocation would cost 32 MB times the number of pinned shapes for no benefit.
torch::Tensor& workspace(const torch::TensorOptions& opts) {
    static torch::Tensor ws;
    if (!ws.defined()) ws = torch::empty({32 * 1024 * 1024}, opts.dtype(torch::kUInt8));
    return ws;
}
size_t workspace_bytes() { return 32u * 1024u * 1024u; }

}  // namespace

// One pinned GEMM: fixed shape, fixed dtype pair, fixed algorithm.
class PinnedGemm {
public:
    PinnedGemm(int64_t m, int64_t n, int64_t k,
               torch::ScalarType a_dtype, torch::ScalarType b_dtype, bool fast_accum,
               bool accum = false, bool nt = false)
        : m_(m), n_(n), k_(k), accum_(accum), nt_(nt) {

        LT_CHECK(cublasLtCreate(&lt_));
        LT_CHECK(cublasLtMatmulDescCreate(&desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t op_a = nt_ ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t op_b = nt_ ? CUBLAS_OP_T : CUBLAS_OP_N;
        LT_CHECK(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_TRANSA,
                                                &op_a, sizeof(op_a)));
        LT_CHECK(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_TRANSB,
                                                &op_b, sizeof(op_b)));
        // int8_t, not bool/int -- the attribute's declared width. Set on the descriptor, so it
        // is in force for the heuristic query and every timing launch, not just the final run.
        const int8_t fa = fast_accum ? 1 : 0;
        LT_CHECK(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                                &fa, sizeof(fa)));
        // Operands swapped (see file header): Adesc describes b, Bdesc describes a. Under nt
        // each stored shape flips with its op, so op(A)/op(B) are the same matrices either way.
        if (nt_) {
            LT_CHECK(cublasLtMatrixLayoutCreate(&Adesc_, fp8_type(b_dtype), n, k, n));
            LT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc_, fp8_type(a_dtype), m, k, m));
        } else {
            LT_CHECK(cublasLtMatrixLayoutCreate(&Adesc_, fp8_type(b_dtype), k, n, k));
            LT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc_, fp8_type(a_dtype), k, m, k));
        }
        LT_CHECK(cublasLtMatrixLayoutCreate(&Ddesc_, accum_ ? CUDA_R_32F : CUDA_R_16BF, n, m, n));
    }

    PinnedGemm(const PinnedGemm&) = delete;
    PinnedGemm& operator=(const PinnedGemm&) = delete;

    ~PinnedGemm() {
        if (Ddesc_) cublasLtMatrixLayoutDestroy(Ddesc_);
        if (Bdesc_) cublasLtMatrixLayoutDestroy(Bdesc_);
        if (Adesc_) cublasLtMatrixLayoutDestroy(Adesc_);
        if (desc_) cublasLtMatmulDescDestroy(desc_);
        if (lt_) cublasLtDestroy(lt_);
    }

    // Enumerate every candidate, screen, then score the finalists fairly. Returns the winner's
    // config plus the heuristic's own pick measured in the same pass, so the caller can log a
    // ratio that means something.
    //
    // Two phases, because a single timed pass in heuristic order is not a fair comparison: it
    // times candidate 0 -- cuBLAS's own pick -- first, on a card still at its ~2460 MHz cold
    // boost, and every later candidate at a lower clock. That biases both the reported ratio and
    // the selection toward candidate 0, which is what the first version of this did.
    std::map<std::string, double> autotune(torch::Tensor a, torch::Tensor b,
                                           torch::Tensor a_scale, torch::Tensor b_scale,
                                           int64_t max_algos, int64_t iters,
                                           int64_t finalists, int64_t rounds) {
        check_operands(a, b);
        cublasLtMatmulPreference_t pref = nullptr;
        size_t ws_bytes = workspace_bytes();
        LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
        std::vector<cublasLtMatmulHeuristicResult_t> results(max_algos);
        int returned = 0;
        LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt_, desc_, Adesc_, Bdesc_, Ddesc_, Ddesc_, pref,
                                                int(max_algos), results.data(), &returned));
        cublasLtMatmulPreferenceDestroy(pref);
        TORCH_CHECK(returned > 0, "cuBLASLt returned no algorithms for this shape");

        // Accum plans time with beta=1 into a scratch buffer (zeroed: garbage fp32 could hold
        // NaN, and the values grow across timing launches, which is harmless).
        auto out = accum_ ? torch::zeros({m_, n_}, a.options().dtype(torch::kFloat32))
                          : torch::empty({m_, n_}, a.options().dtype(torch::kBFloat16));

        // Phase 1, screen. Time every candidate cheaply, in heuristic order. These numbers are
        // biased cool-to-hot and are used only to rank, never as the verdict -- the pass exists
        // mostly to be the soak: a few hundred back-to-back GEMMs of this exact shape is what
        // takes the card off its cold boost, and soaking on the shape being scored is what
        // probe_cublaslt.py --equilibrium found to matter.
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
        TORCH_CHECK(!screened.empty(), "no cuBLASLt algorithm ran successfully for this shape");

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
                times[i].push_back(
                    time_algo(results[i].algo, a, b, a_scale, b_scale, out, iters));
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
        // What the old single-pass protocol would have reported, so the bias stays visible.
        info["screen_us"] = screen_us[best_i];
        info["screen_heuristic_us"] = screen_us[heuristic_i];
        info["finalists"] = double(pool.size());
        info["rounds"] = double(rounds);
        info["tile"] = get_cfg(algo_, CUBLASLT_ALGO_CONFIG_TILE_ID);
        info["splitk"] = get_cfg(algo_, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
        info["reduction"] = get_cfg(algo_, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
        info["stages"] = get_cfg(algo_, CUBLASLT_ALGO_CONFIG_STAGES_ID);
        info["algo_id"] = get_cfg(algo_, CUBLASLT_ALGO_CONFIG_ID);
        info["swizzle"] = get_cfg(algo_, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
        info["waves"] = results[best_i].wavesCount;
        return info;
    }

    torch::Tensor run(torch::Tensor a, torch::Tensor b,
                      torch::Tensor a_scale, torch::Tensor b_scale) {
        TORCH_CHECK(pinned_, "PinnedGemm.run() before autotune()");
        TORCH_CHECK(!accum_, "accum plan: use run_accum()");
        check_operands(a, b);
        auto out = torch::empty({m_, n_}, a.options().dtype(torch::kBFloat16));
        auto st = launch(algo_, a, b, a_scale, b_scale, out);
        TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "pinned cublasLtMatmul failed: ", int(st));
        return out;
    }

    // out += a.b (scaled): beta=1 with C = D = out, the caller's fp32 grad accumulator.
    void run_accum(torch::Tensor a, torch::Tensor b,
                   torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor out) {
        TORCH_CHECK(pinned_, "PinnedGemm.run_accum() before autotune()");
        TORCH_CHECK(accum_, "non-accum plan: use run()");
        check_operands(a, b);
        TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32 && out.dim() == 2 &&
                    out.size(0) == m_ && out.size(1) == n_ && out.is_contiguous(),
                    "out must be a contiguous [m, n] fp32 CUDA tensor");
        auto st = launch(algo_, a, b, a_scale, b_scale, out);
        TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "pinned accum cublasLtMatmul failed: ", int(st));
    }

private:
    void check_operands(const torch::Tensor& a, const torch::Tensor& b) const {
        TORCH_CHECK(a.is_cuda() && b.is_cuda(), "operands must be CUDA tensors");
        if (nt_) {
            // Natural layouts: a = go as [k,m] row-major, b = x as [k,n] row-major.
            TORCH_CHECK(a.dim() == 2 && a.size(0) == k_ && a.size(1) == m_,
                        "a has the wrong shape (nt)");
            TORCH_CHECK(b.dim() == 2 && b.size(0) == k_ && b.size(1) == n_,
                        "b has the wrong shape (nt)");
            TORCH_CHECK(a.stride(1) == 1 && a.stride(0) == m_, "a must be row-major contiguous");
            TORCH_CHECK(b.stride(1) == 1 && b.stride(0) == n_, "b must be row-major contiguous");
            return;
        }
        // a: logical [m,k] row-major. b: logical [k,n] column-major.
        TORCH_CHECK(a.dim() == 2 && a.size(0) == m_ && a.size(1) == k_, "a has the wrong shape");
        TORCH_CHECK(b.dim() == 2 && b.size(0) == k_ && b.size(1) == n_, "b has the wrong shape");
        TORCH_CHECK(a.stride(1) == 1 && a.stride(0) == k_, "a must be row-major contiguous");
        TORCH_CHECK(b.stride(0) == 1 && b.stride(1) == k_, "b must be column-major");
    }

    cublasStatus_t launch(const cublasLtMatmulAlgo_t& algo, const torch::Tensor& a,
                          const torch::Tensor& b, const torch::Tensor& a_scale,
                          const torch::Tensor& b_scale, torch::Tensor& out) {
        const float alpha = 1.0f, beta = accum_ ? 1.0f : 0.0f;
        // Scales follow their operands through the swap.
        const float* a_sp = a_scale.data_ptr<float>();
        const float* b_sp = b_scale.data_ptr<float>();
        LT_CHECK(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                &b_sp, sizeof(b_sp)));
        LT_CHECK(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                &a_sp, sizeof(a_sp)));
        auto& ws = workspace(a.options());
        return cublasLtMatmul(lt_, desc_, &alpha, b.data_ptr(), Adesc_, a.data_ptr(), Bdesc_,
                              &beta, out.data_ptr(), Ddesc_, out.data_ptr(), Ddesc_, &algo,
                              ws.data_ptr(), workspace_bytes(), at::cuda::getCurrentCUDAStream());
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

    int64_t m_, n_, k_;
    bool accum_ = false;
    bool nt_ = false;
    bool pinned_ = false;
    cublasLtHandle_t lt_ = nullptr;
    cublasLtMatmulDesc_t desc_ = nullptr;
    cublasLtMatrixLayout_t Adesc_ = nullptr, Bdesc_ = nullptr, Ddesc_ = nullptr;
    cublasLtMatmulAlgo_t algo_{};
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<PinnedGemm>(m, "PinnedGemm")
        .def(pybind11::init<int64_t, int64_t, int64_t, torch::ScalarType, torch::ScalarType,
                            bool, bool, bool>(),
             pybind11::arg("m"), pybind11::arg("n"), pybind11::arg("k"), pybind11::arg("a_dtype"),
             pybind11::arg("b_dtype"), pybind11::arg("fast_accum"),
             pybind11::arg("accum") = false, pybind11::arg("nt") = false)
        .def("autotune", &PinnedGemm::autotune)
        .def("run", &PinnedGemm::run)
        .def("run_accum", &PinnedGemm::run_accum);
}
