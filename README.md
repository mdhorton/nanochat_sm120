## NOTE: This is a fork of the nanochat repo: https://github.com/karpathy/nanochat

# nanochat for sm120 GPUs

Nanochat targets datacenter H100 (sm90) GPUs. This fork targets consumer/workstation Blackwell GPUs (sm120). These sm120
GPUs will no doubt be slower. However, some questions I was curious about:

- Are there blackwell specific features that can increase performance?
- How does H100 performance compare against RTX Pro 6000 Blackwell?
- How does the rental cost compare? (eg, cheaper cost per hour but more hours are needed)

All benchmarks are Blackwell GPUs, except the H100s.

# shortrun.sh

These are short test runs that complete in under ~15 minutes (usually between 5-10 minutes). Full runs are performed
later for more complete validation.

## --depth 24 --device-batch-size 16 --num-iterations 50

toks/USD: 171M

| GPUs        | toks/sec | mem GB |      bpb | USD/hour |
|-------------|---------:|-------:|---------:|---------:|
| 2x H100 SXM |      244 |   57.2 | 1.330220 |     5.00 |
| 4x H100 SXM |      485 |   54.7 | 1.330113 |    10.00 |
| 8x H100 SXM |          |        |          |    20.00 |

| GPUs               | toks/sec | mem GB |      bpb | USD/hour |
|--------------------|---------:|-------:|---------:|---------:|
| 2x RTX Pro 6000 WS |      130 |     57 | 1.328198 |     2.75 |
| 4x RTX Pro 6000 WS |          |        |          |          |
| 8x RTX Pro 6000 WS |      248 |     53 | 1.324561 |    10.00 |

### 2x RTX Pro 6000 WS

USD cost/hour: 3.00
best toks/USD: 155M

| toks/sec | mem GB |      bpb | notes                    |
|---------:|-------:|---------:|--------------------------|
|      71k |   64.8 | 1.576989 | baseline (bf16)          |
|      79k |   57.2 | 1.566846 | --fp8                    |
|     107k |   64.8 | 1.590402 | --window-pattern L       |
|     129k |   57.2 | 1.581414 | --fp8 --window-pattern L |

## --depth 12 --device-batch-size 8 --num-iterations 50

These GPUs have less VRAM so depth and device-batch-size must be reduced. This increases training speed, but at the cost
of model quality.

### 2x RTX Pro 4000

USD cost/hour: 0.50
best toks/USD: 1.41B

| toks/sec | mem GB |      bpb | flags                                           |
|---------:|-------:|---------:|-------------------------------------------------|
|     112k |   10.2 | 1.676463 |                                                 |
|     120k |   11.4 | 1.676669 | --fp8                                           |
|     164k |   10.2 | 1.685996 | --window-pattern L                              |
|     185k |   11.4 | 1.686771 | --fp8 --window-pattern L                        |
|     173k |   10.2 | 1.676523 | windowed flash (SSSL)                           |
|     195k |   11.4 | 1.675813 | --fp8 + windowed flash (SSSL)                   |
|     162k |    8.8 | 1.688276 | --nvfp4 --window-pattern L                      |
|     164k |   10.7 | 1.692122 | --nvfp4 --nvfp4-lm-head --window-pattern L      |
|     156k |    9.5 | 1.687710 | --nvfp4 --nvfp4-no-bwd-quant --window-pattern L |
|     170k |    9.0 | 1.679255 | --nvfp4 + windowed flash (SSSL)                 |

The first two rows predate windowed flash: their `SSSL` layers ran through an SDPA mask.

### 4x RTX Pro 4000

USD cost/hour: 1.00
best toks/USD: 1.11B

| toks/sec | mem GB |      bpb | flags                    |
|---------:|-------:|---------:|--------------------------|
|     198k |    9.6 | 1.672383 | baseline (bf16)          |
|     211k |   10.8 | 1.677822 | --fp8                    |
|     278k |    9.6 | 1.682599 | --window-pattern L       |
|     307k |   10.8 | 1.687786 | --fp8 --window-pattern L |

# nvfp4 (Quartet II)

`--nvfp4` trains the transformer linears in NVFP4 using the Quartet II recipe (RTN + 4/6 forward,
RHT + MS-EDEN backward), torch-native: `torch._scaled_mm` for the FP4 GEMMs and Triton quantizers
ported from [Quartet-II](https://github.com/IST-DASLab/Quartet-II) (`nanochat/nvfp4_kernels.py`,
`nanochat/nvfp4.py`). Requires Blackwell (sm100+); torch.compile is mandatory (eager is 3-5x slower).
Flags: `--nvfp4-lm-head` (also quantize lm_head), `--nvfp4-no-bwd-quant` (bf16 backward).
Microbench (`python -m scripts.nvfp4_bench --compile`): at d24 shapes nvfp4 fwd+bwd beats bf16 by
1.1-1.6x (ahead of `--fp8`); at d12 the 768-wide GEMMs are too small and it only reaches bf16 parity
end-to-end, with ~15% lower peak memory (4-bit saved activations).

# windowed flash attention

https://github.com/facebookresearch/xformers

FA3 has no sm120 build, so the `SSSL` sliding-window default used to fall back to an explicit SDPA mask (4.7x the cost
of a flash forward). `nanochat/sm120/attention.py` routes S layers through `aten::_flash_attention_forward` with
`window_size_left` instead, so `SSSL` now beats `--window-pattern L` by ~5% at identical memory.
`python -m scripts.probe_attention` times the three arms; `tests/test_attention_window.py` checks the kernel against
the mask path.

## License

MIT
