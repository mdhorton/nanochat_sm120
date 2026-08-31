
2026-08-30 
baseline           120k toks/s
--window-pattern L 185k toks/s

2026-08-31 windowed flash (nanochat/sm120/attention.py), d12 dbs8, 2x RTX Pro 4000
--fp8 --window-pattern L      186k toks/s
--fp8 SSSL (windowed flash)   195k toks/s  (+5%; was 120k via SDPA mask)
bf16: L 165k, SSSL 173k
