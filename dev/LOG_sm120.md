
2026-08-30 
baseline           120k toks/s
--window-pattern L 185k toks/s

2026-08-30: NVFP4 (Quartet II) landed: nanochat/nvfp4_kernels.py + nvfp4.py, --nvfp4 in base_train.
shortrun d12 2xRTX Pro 4000 (all with -L): nvfp4 162k tok/s / 8.8GB / bpb 1.6883 vs bf16 164k/10.2/1.6860,
fp8 185k/11.4/1.6868. d12 GEMMs too small to win; compiled microbench at d24 shapes: nvfp4 1.1-1.6x over
bf16 fwd+bwd, ahead of fp8. Fused Triton EDEN kernels (transposed-read + requant) were the key perf fix.

2026-08-30 FlexAttention (2x RTX Pro 4000, d12 dbs8, 50 iters, all --fp8)
Without FA3, SDPA implements a sliding window with a dense TxT mask, which is 6x slower than no window
at all. --attn-impl flex replaces it with a BlockMask (72.7% of 128x128 blocks skipped at window=512).

--window-pattern L                  185k toks/s  bpb 1.687199
--window-pattern SSSL               121k toks/s  bpb 1.676072
--window-pattern SSSL --attn-impl flex  187k toks/s  bpb 1.675918

So SSSL is now free: 1.55x faster than the SDPA windowed path, and slightly faster than not windowing,
while keeping SSSL's better bpb. Per-layer kernel times (B=8 H=10 T=2048 D=128, fwd+bwd, see
dev/bench_attention.py): SDPA causal 3.82ms, SDPA dense mask 23.62ms, flex causal 5.74ms, flex
sliding 2.97ms. flex loses on dense causal, so full-context layers stay on SDPA/FA3.
