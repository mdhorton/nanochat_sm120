# NVFP4 recipe: horizon arm plan

Does `--nvfp4-exclude lm_head --nvfp4-exclude-precision fp8-fwd` close the +0.0137 bpb deficit
C2 measured (dev/nvfp4-quartet.md, *Numerics: C2*), and does NVIDIA's 80/20 schedule
(`--warmdown-ratio 0.2`) change the answer? 100-step runs cannot tell: their seed noise is
±0.013, the size of the effect. The C2 protocol can: at 2,520 steps the fp8 seeds agreed to
4e-6 and the nvfp4 seeds to 7e-5, so any paired delta above ~0.001 is real.

## Where

This host (2x RTX PRO 4000): d12 / dbs 8 / ratio 12, ~2 h per arm, the config C2 was measured on.
The d24 question is separate (see *Then d24*).

## Arms

Base, identical to C2: `--depth 12 --device-batch-size 8 --target-param-data-ratio 12 --seed 42`.
Seed 42 pairs every new arm with the existing C2 checkpoints.

| arm                         | flags                                                               | status                                     |
|-----------------------------|---------------------------------------------------------------------|--------------------------------------------|
| fp8, warmdown 0.65          | `--fp8`                                                             | done: c2-fp8-s42 0.833421 (s43 0.833425)   |
| nvfp4 plain, 0.65           | `--nvfp4`                                                           | done: c2-nvfp4-s42 0.847110 (s43 0.847187) |
| nvfp4 recipe, 0.65          | `--nvfp4 --nvfp4-exclude lm_head --nvfp4-exclude-precision fp8-fwd` | done: c5-nvfp4-recipe-w65 0.844619         |
| fp8, 0.2                    | `--fp8 --warmdown-ratio 0.2`                                        | done: c5-fp8-w20 0.840602                  |
| nvfp4 recipe, 0.2           | recipe flags + `--warmdown-ratio 0.2`                               | done: c5-nvfp4-recipe-w20 0.853487         |
| fp8, 0.65 (control)         | `--fp8`                                                             | done: c5-fp8-w65 0.833176                  |
| nvfp4 plain, 0.65 (control) | `--nvfp4`                                                           | done: c5-nvfp4-plain-w65 0.846993          |

Run 2026-09-03 as five arms, 8.7 h: the two 0.65 controls were added because C2 was measured in
the sibling checkout (`sm120_nanochat`), so its constants are a cross-tree reference. They
reproduced it — plain deficit +0.013817 against C2's +0.013689. Numerics-only, so no `--cooldown`.
The C2 nvfp4 arm ran without the weight cache / fused wgrad stack; those change timing, not
numerics, and all five c5 arms carry them uniformly.

Three corrections to the command below were needed and are folded in: `--base-dir` (else the run
falls back to `~/.cache/nanochat`), `--eval-tokens 2097152` (C2 pinned it; the default is 20x the
eval work *and a different estimator*), and dropping `--run` (wandb is unauthenticated here).
`NANOCHAT_FA2_SWINDOW=1` must be exported: without it 3 of 4 layers fall back to SDPA mask
emulation, which is 1.6x slower *and* changes the arithmetic.

```
NANOCHAT_FA2_SWINDOW=1 scripts/arm_batch.sh --out dev-ignore/overnight/c5 \
  --base-dir /remote/.nanochat-cache \
  --base "--depth 12 --device-batch-size 8 --target-param-data-ratio 12 --seed 42 --total-batch-size 524288 --eval-every 250 --eval-tokens 2097152 --core-metric-every -1 --sample-every -1" -- \
  fp8-w65          "--fp8 --model-tag c5-fp8-w65" \
  nvfp4-plain-w65  "--nvfp4 --model-tag c5-nvfp4-plain-w65" \
  nvfp4-recipe-w65 "--nvfp4 --nvfp4-exclude lm_head --nvfp4-exclude-precision fp8-fwd --model-tag c5-nvfp4-recipe-w65" \
  fp8-w20          "--fp8 --warmdown-ratio 0.2 --model-tag c5-fp8-w20" \
  nvfp4-recipe-w20 "--nvfp4 --nvfp4-exclude lm_head --nvfp4-exclude-precision fp8-fwd --warmdown-ratio 0.2 --model-tag c5-nvfp4-recipe-w20"
```

Progress: `dev-ignore/overnight/c5/progress.txt`; final bpb:
`grep "Minimum validation bpb" dev-ignore/overnight/c5/*.log`.

## What each pair answers

Answered within the batch rather than against C2, since the controls ran.

| pair                      | question                                     | answer                              |
|---------------------------|----------------------------------------------|-------------------------------------|
| recipe 0.65 vs fp8 0.65   | how much of the +0.0137 the recipe closes    | **+0.011443** — 17% closed          |
| recipe 0.65 vs plain 0.65 | the recipe's gain in isolation               | **-0.002374**, flat across training |
| fp8 0.2 vs fp8 0.65       | what 80/20 does to fp8 alone at this horizon | **+0.007426** — a loss              |
| recipe 0.2 vs fp8 0.2     | the deficit under NVIDIA's schedule          | **+0.012885** — worse than at 0.65  |

The per-eval trajectory (eval every 250 steps) settled which of the two it is: **the recipe only
shifts the curve.** Its gain runs 0.0036 / 0.0026 / 0.0027 / 0.0020 / 0.0022 at steps 500-2500
while the deficit beneath widens at plain NVFP4's rate, so it removes a constant offset rather
than the compounding systematic.

## Decision — resolved 2026-09-03, third branch

- ~~Recipe deficit under 0.003 bpb at either schedule: adopt, then confirm at d24.~~
- ~~Deficit 0.003-0.010: next candidates, one per batch.~~
- **Deficit +0.011443 at w65 and +0.012885 at w20 — the third branch.** The recipe closes 17% of
  C2's +0.0137 and its gain is flat across training (0.0036 -> 0.0022 from step 500 to 2500) while
  the deficit underneath widens at plain NVFP4's rate. It removes a constant offset, not the
  compounding systematic. **Forward noise in the blocks, not lm_head, is the cost at d12**;
  value-level stochastic rounding (CUDA) is the remaining lever.
- Of the queued candidates only `--nvfp4-bf16-blocks 0,1` still earns an arm — it attacks the
  block forwards, where C5 says the cost is. `--nvfp4-scaling delayed` and `--nvfp4-bwd-source
  bf16` are backward-side; C5 gives no reason to expect either to move a forward-noise deficit.
- Keep the recipe flags anyway: -0.0024 bpb for +1.9% wall clock (95.83m vs 94.05m).
- `--warmdown-ratio 0.2` is a **loss** at ratio 12, for both precisions (fp8 +0.0074, recipe
  +0.0089), and widens the deficit to +0.0129. Both w20 arms are still descending steeply at the
  last eval, so it is a schedule for a longer horizon than this one.

## Then d24

lm_head is a quarter of d12's linear compute and much less of d24's, so the recipe's gain should
*shrink* with depth while the block-forward deficit C5 points at grows. That makes d24 a weaker
case for the recipe, not a stronger one, and argues for spending the RTX PRO 6000 box on
stochastic rounding rather than on a d24 confirmation of a 17% partial fix. The 100-step d24 pair
(fp8 1.302880, nvfp4 1.288498, both warmdown 0.2) is inside noise and decides nothing; C5's w20
result now also contradicts the schedule it used.
