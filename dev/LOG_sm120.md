
2026-08-30 
baseline           120k toks/s
--window-pattern L 185k toks/s

2026-08-30: NVFP4 (Quartet II) landed: nanochat/nvfp4_kernels.py + nvfp4.py, --nvfp4 in base_train.
shortrun d12 2xRTX Pro 4000 (all with -L): nvfp4 162k tok/s / 8.8GB / bpb 1.6883 vs bf16 164k/10.2/1.6860,
fp8 185k/11.4/1.6868. d12 GEMMs too small to win; compiled microbench at d24 shapes: nvfp4 1.1-1.6x over
bf16 fwd+bwd, ahead of fp8. Fused Triton EDEN kernels (transposed-read + requant) were the key perf fix.

2026-08-31 windowed flash (nanochat/sm120/attention.py), d12 dbs8, 2x RTX Pro 4000
--fp8 --window-pattern L      186k toks/s
--fp8 SSSL (windowed flash)   195k toks/s  (+5%; was 120k via SDPA mask)
bf16: L 165k, SSSL 173k
