
2026-08-30 
baseline           120k toks/s
--window-pattern L 185k toks/s

2026-08-31 windowed flash (nanochat/sm120/attention.py), d12 dbs8, 2x RTX Pro 4000
--fp8 --window-pattern L      186k toks/s
--fp8 SSSL (windowed flash)   195k toks/s  (+5%; was 120k via SDPA mask)
bf16: L 165k, SSSL 173k

2026-08-31 windowed flash made opt-in (NANOCHAT_FA2_WINDOWED_FLASH=1)
Default is now upstream-faithful: no install, S layers take the SDPA mask.
The install moved out of module scope, so importing nanochat.sm120.nvfp4 (i.e. --nvfp4)
no longer enables it as a side effect. Zero changes to upstream-owned files -- base_train's
existing `_windowed_impl is None` predicates already express the switch.

See dev/attention-ideas.md for what attention work is ruled out (FlexAttention, document/varlen
masking) and what is still open (attention sinks, score softcap, the q/k 1.2 scale).
