# Attention: what is ruled out, what is still open

Written 2026-08-31 to stop re-litigating FlexAttention. The evidence was scattered across the
`flex-attention` branch, README's *Rejected* section, and upstream's `dev/LOG.md`.

**Throughput: there is nothing left to take from FlexAttention.** Its only measured win was over the
SDPA dense-mask fallback, and `NANOCHAT_FA2_WINDOWED_FLASH=1` deleted that fallback and beats it.

## Ruled out

| idea | verdict | evidence |
|---|---|---|
| Flex for sliding-window (`S`) layers | superseded | 187k tok/s vs windowed flash's 195k, bpb 1.675918 vs 1.675813 — README *Rejected* (d12 dbs8, 2x RTX PRO 4000, `--fp8`). Flex also requires `torch.compile` and cannot run eval eagerly |
| Flex for dense-causal (`L`) layers | slower | 5.74 ms vs SDPA causal 3.82 ms fwd+bwd, `dev/bench_attention.py` on branch `flex-attention`. That branch excluded `L` layers itself |
| Attention sinks for `S` layers | no pathology, and the sink is already in window | probed at 8 checkpoints from step 50 to a converged 2520 (val_bpb 0.847), `scripts/probe_attention_sinks.py`. See *Attention sinks* below |
| Document / varlen masking | null result | upstream `dev/LOG.md`, "2026-01-13: Varlen Attention (Negative Result)", d16: val_bpb 0.85427 -> 0.85407, MFU and tok/s "~same". Gotchas recorded there: variable `cu_seqlens` caused 25 s/iter recompiles, and `nonzero()` inside the compiled region hit the recompile limit |

## Why document masking looks better than it measures

The corpus statistics are genuinely suggestive, which is the trap. Measured 2026-08-31 against
`base_data_climbmix` with this repo's tokenizer (**this session, not a recorded experiment**):

- mean doc 640 tokens, median ~530; **96.4% shorter than `max_seq_len` 2048**; **~3.2 docs per row**;
  **48% shorter than the 512-token `S` window**. A training row is almost never one document.
- Cross-document pairs are **50.4%** of `L`-layer and **27.7%** of `S`-layer attended pairs.

Re-measured on packed rows: **3.26 BOS per row**, cross-document pairs **60.9%** (`L`) and **26.1%**
(`S`), and a BOS in window for **59.5%** of `S`-layer query positions past 512. The original figures
stand.

**Gotcha when measuring any of this:** discard the first ~24 batches. `refill_buffer` starts with an
empty `doc_buffer`, so best-fit has few candidates and packs one long document per row; the first
document averages 1556 tokens and BOS-in-window sits at 6.5%, against 852 and ~60% once the buffer
fills toward `buffer_size=1000`. Measuring the first few batches gets you numbers that are wrong by
2-10x and look plausible. `scripts/probe_attention_sinks.py --skip-batches` exists for this.

Three reasons the prize is over-stated:

- Attention is only **13.8% of the step** (501.3 of 3639.8 ms, d20 kernel decomposition in the
  sibling's `dev/nvfp4-quartet.md`), so the blended ceiling is ~5% of wall clock.
- Pair count over-predicts by ~2x. From the sibling's `dev/perf-log-experiments.md`: *"the window is
  worth 0.616x, not the 0.437x its attended-pair count implies — the backward carries `dot_do_o` and
  `convert_dq` passes that scale with sequence, not with the window."*
- Flash works in 128x128 blocks and boundaries land mid-block, so realized savings are lower again.

Net expected: below this repo's own 2% "not evidence" bar — consistent with the measured null.

## Still open

Leads with their cost, not recommendations.

- **Attention-score softcap.** Never tried. Distinct from the *lm_head logit* softcap the model
  already has (`nanochat/gpt.py:511-515`, `softcap=15`, tuned in upstream's log). Needs flex or a
  kernel with a softcap parameter.
- **The hardcoded attention temperature.** `nanochat/gpt.py:103-104` does `q *= 1.2; k *= 1.2`
  (scores x1.44) with `TODO think through better`. The score-shaping lever already present, never
  swept, free to try.
- **Windowed inference / chunked prefill.** `flash_attn_with_kvcache` still uses mask emulation when
  `Tq != Tk`; the sibling's `dev/perf-log.md` names it as the open follow-up. Latency, not training
  throughput. Flex does not help — eager flex materializes the score matrix.

## Attention sinks: probed and ruled out

Probed 2026-08-31 with `scripts/probe_attention_sinks.py`, which patches `flash_attn.flash_attn_func`
and recomputes the softmax exactly in fp32 for 128 query positions >= 512. Eight checkpoints, from
step 50 to a purpose-trained converged d12 SSSL run (2520 steps, val_bpb 0.847, 94 min on 2x RTX PRO
4000). Two questions, both answered no.

**Is there a moving-edge pathology?** No, and not at any point in training. Mass on the oldest
in-window key stayed at **0.1-0.9x uniform** — *below* what uniform attention would put there — across
a 6x range of training and a bpb from 1.68 to 0.847. It never once rose. Over the same span `L`-layer
sink demand grew ~25x (layer 7: 2.1x uniform at step 50 to 233x at 2520), so this is not a model that
had yet to develop an appetite.

**Would a permanent sink help anyway?** The appetite is real but already fed. The sink this
architecture forms is a **content** sink on the BOS token, not a positional one: at convergence layer
7 puts **36.8%** of its mass on BOS tokens and layer 11 **26.7%**, and the value vectors there
collapse to **0.18-0.44** of the mean norm — the signature of a no-op probability dump. `S` layers do
the same thing when they can: layer 9 puts **11.6%** of its mass on in-window BOS at a value norm of
**0.24**. And they usually can — packing puts a BOS in the 512 window for **~60-67%** of query
positions past 512.

That leaves the residual case: the ~35% of positions with no BOS in window. Those show higher
attention entropy (layer 9: 3.61 nats with a sink in window, 4.28 without), but the comparison is
confounded — queries with a BOS in window are near a document start, so their current-document
context is short and concentrated attention is expected mechanically. Against that uncertain and
partial upside, the only exact route is flex `mask_mod = causal & (in-window | kv_idx == 0)` at ~4%
tok/s, training-only. Paying a measured 4% throughput for a gain that is probably below this repo's
2% "not evidence" bar is a bad trade. Dropped.

The LSE route stays blocked for the reason recorded below: a learned per-head sink logit as
`out * sigmoid(lse - s)` is exact and nearly free in the forward, but `_flash_attention_backward`
takes no `grad_lse`.

## Two facts worth not rediscovering

- `aten::_flash_attention_forward` **already accepts `cu_seqlens`** — the two `None`s at
  `nanochat/sm120/attention.py:50-52` — and `_flash_attention_backward` takes the same pair. Revisiting
  varlen needs no new kernel, only boundaries plumbed through (the dataloader currently discards them:
  `nanochat/dataloader.py` packs by best-fit and keeps no offsets).
- That op has **no sink and no softcap parameter** in torch 2.9, which is why both ideas above cost
  either flex or a custom kernel. It does expose `alibi_slopes` — irrelevant here, the model uses RoPE.
