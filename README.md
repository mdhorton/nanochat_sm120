## nanochat fork for sm120 GPUs

**NOTE: This is a fork of the nanochat repo:** https://github.com/karpathy/nanochat

Nanochat targets datacenter H100 (sm90) GPUs. This fork targets consumer/workstation Blackwell GPUs (sm120). These sm120
GPUs will no doubt be slower. However, some questions I was curious about:

- Are there blackwell specific features that can increase performance?
- How does H100 performance compare against RTX Pro 6000 Blackwell?
- How does the rental cost compare? (eg, cheaper cost per hour but more hours are needed)

This fork is a learning exercise.

Most of the benchmarks were run with either 2 or 4 GPUs. This was due to the fact that 8 GPUs were harder to find. 4x
GPUs generally ran about twice as fast as 2x, but cost twice as much. Overll it was a wash in terms of total cost.

## shortrun.sh

These are short test runs that take ~5-10 minutes. Full runs were performed later for more complete validation.

# --depth 24 --device-batch-size 16 --num-iterations 50

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

# --depth 12 --device-batch-size 8 --num-iterations 50

These GPUs have less VRAM so depth and device-batch-size must be reduced. This increases training speed, but at the cost
of model quality.

### 2x RTX Pro 4000

USD cost/hour: 0.50
best toks/USD: 1.33B

| toks/sec | mem GB |      bpb | notes                    |
|---------:|-------:|---------:|--------------------------|
|     112k |   10.2 | 1.676463 | baseline (bf16)          |
|     120k |   11.4 | 1.676669 | --fp8                    |
|     164k |   10.2 | 1.685996 | --window-pattern L       |
|     185k |   11.4 | 1.686771 | --fp8 --window-pattern L |

### 4x RTX Pro 4000

USD cost/hour: 1.00
best toks/USD: 1.11B

| toks/sec | mem GB |      bpb | notes                    |
|---------:|-------:|---------:|--------------------------|
|     198k |    9.6 | 1.672383 | baseline (bf16)          |
|     211k |   10.8 | 1.677822 | --fp8                    |
|     278k |    9.6 | 1.682599 | --window-pattern L       |
|     307k |   10.8 | 1.687786 | --fp8 --window-pattern L |

## License

MIT
