
2026-08-30 
baseline           120k toks/s
--window-pattern L 185k toks/s

2026-08-30: NVFP4 (Quartet II) landed: nanochat/nvfp4_kernels.py + nvfp4.py, --nvfp4 in base_train.
shortrun d12 2xRTX Pro 4000 (all with -L): nvfp4 162k tok/s / 8.8GB / bpb 1.6883 vs bf16 164k/10.2/1.6860,
fp8 185k/11.4/1.6868. d12 GEMMs too small to win; compiled microbench at d24 shapes: nvfp4 1.1-1.6x over
bf16 fwd+bwd, ahead of fp8. Fused Triton EDEN kernels (transposed-read + requant) were the key perf fix.
