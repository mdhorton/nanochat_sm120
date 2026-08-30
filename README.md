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

These are short test runs that take ~5 minutes. Full runs were performed later for more complete validation.

> --depth 24 --device-batch-size 16 --num-iterations 50

| GPUs        | toks/sec | peak mem GB | final bpb | $/hour |
|-------------|---------:|------------:|----------:|-------:|
| 2x H100 SXM |      244 |          57 |  1.330220 |   5.00 |
| 4x H100 SXM |      485 |          54 |  1.330113 |  10.00 |
| 8x H100 SXM |          |             |           |  20.00 |

| GPUs               | toks/sec | peak mem GB | final bpb | $/hour |
|--------------------|---------:|------------:|----------:|-------:|
| 2x RTX Pro 6000 WS |      130 |          57 |  1.328198 |   2.75 |
| 4x RTX Pro 6000 WS |          |             |           |        |
| 8x RTX Pro 6000 WS |      248 |          53 |  1.324561 |  10.00 |

| 2x R | vs baseline | mem | bpb      | notes              |
|------|-------------|-----|----------|--------------------|
| 340k |             | 11G | 1.468449 | baseline           |
| 610k | 1.79x       | 11G | 1.479908 | --window-pattern L |

| 2x RTX Pro 6000 WS | vs baseline | mem | bpb      | notes              |
|--------------------|-------------|-----|----------|--------------------|
| 340k               |             | 11G | 1.468449 | baseline           |
| 610k               | 1.79x       | 11G | 1.479908 | --window-pattern L |

| 2x RTX Pro 4000 | vs baseline | | bpb | notes              |
|-----------------|-------------|-|-----|--------------------|
| 120k            |             | |     | baseline           |
| 185k            |             | |     | --window-pattern L |

## License

MIT
