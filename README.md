## nanochat fork for sm120 GPUs

**NOTE: This is a fork of the nanochat repo:** https://github.com/karpathy/nanochat

Nanochat targets datacenter H100 (sm90) GPUs. This fork targets consumer/workstation Blackwell GPUs (sm120). These sm120
GPUs will no doubt be slower. However, the questions I was curious about:

- Are there blackwell specific features that can increase performance?
- How does H100 performance compare against RTX Pro 6000 Blackwell?
- How does the rental cost compare? (eg, cheaper cost per hour but more hours are needed)

This fork is a learning exercise.

## toks/sec

--depth 24
--device-batch-size 16

|             | toks/sec |
|-------------|-----|
| 4x H100 SXM | 485 |
| 2x H100 SXM | 485 | 

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
