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

| toks/sec | mem GB | bpb | notes |
|---------:|-------:|---------:|--------------------------|----------------------------------------------|
| 71k | 64.8 | 1.576989 | baseline (bf16)          |
| 79k | 57.2 | 1.566846 | --fp8 |
| 107k | 64.8 | 1.590402 | --window-pattern L |
| 129k | 57.2 | 1.581414 | --fp8 --window-pattern L |

## --depth 12 --device-batch-size 8 --num-iterations 50

These GPUs have less VRAM so depth and device-batch-size must be reduced. This increases training speed, but at the cost
of model quality.

### 2x RTX Pro 4000

Most of the tests start here because this is my personal workstion.

#### bf16

base flag: none

| toks/sec | mem GB | bpb | extra flags |                   |
|---------:|-------:|----:|-------------|-------------------|
|          |        |     |             |                   |
|          |        |     |             |                   |
|          |        |     |             |                   |

#### fp8

base flag: --fp8

| toks/sec | mem GB |      bpb | extra flags                                         |                   
|---------:|-------:|---------:|-----------------------------------------------------|
|     120k |   11.4 | 1.676669 |                                                     |                   
|     185k |   11.4 | 1.686771 | --window-pattern L                                  |  
|     195k |   11.4 | 1.675813 | NANOCHAT_FA2_WINDOWED_FLASH=1                       |                   
|     205k |    9.7 | 1.662816 | NANOCHAT_FA2_WINDOWED_FLASH=1 --fp8-scaling delayed |                   

#### nvfp4

base flag: --nvfp4

| toks/sec | mem GB |      bpb | extra flags                   |                                          
|---------:|-------:|---------:|-------------------------------|
|     135k |   14.0 | 1.681464 |                               |                                          
|     220k |   14.0 | 1.691275 | --window-pattern L            |                                          
|     234k |   14.0 | 1.681451 | NANOCHAT_FA2_WINDOWED_FLASH=1 |  
|          |        |          |                               |                                          

### 4x RTX Pro 4000

USD cost/hour: 1.00
best toks/USD: 1.11B

| toks/sec | mem GB |      bpb | flags                                                         |
|---------:|-------:|---------:|---------------------------------------------------------------|
|     198k |    9.6 | 1.672383 | baseline (bf16)                                               |
|     211k |   10.8 | 1.677822 | --fp8                                                         |
|     278k |    9.6 | 1.682599 | --window-pattern L                                            |
|     307k |   10.8 | 1.687786 | --fp8 --window-pattern L                                      |
|     320k |   10.8 | 1.677705 | --fp8 NANOCHAT_FA2_WINDOWED_FLASH=1                           |                                          
|     378k |   14.0 | 1.683084 | --nvfp4 NANOCHAT_FA2_WINDOWED_FLASH=1                         | 
|     334k |    9.0 | 1.664606 | --fp8 --fp8-scaling delayed NANOCHAT_FA2_WINDOWED_FLASH=1     | 
|     381k |   14.0 | 1.682999 | --nvfp4 --nvfp4-scaling delayed NANOCHAT_FA2_WINDOWED_FLASH=1 | 

## Accepted

### FA2 sliding-window kernels

NANOCHAT_FA2_WINDOWED_FLASH=1

FA3 has no sm120 kernels, so the `SSSL` sliding-window default uses an explicit SDPA mask, which is very slow. The
`--window-pattern L` option helps but it's not optimal. This new option allows it to use FA2 kernels for the S layers
instead of SDPA masking.

The implementation is mostly just some wiring code so that existing FA2 kernels are used for sliding-windows.

The switch is an environment variable rather than a flag because it has to reach every entry point, including the eval
paths that take no flags. This is an acceptable trade off given that the purpose of this repo is to explore and learn.

## Rejected

### FlexAttention

https://pytorch.org/blog/flexattention/

branch: flex-attention

| toks/sec | mem GB |      bpb | flags                  | 
|---------:|-------:|---------:|------------------------|
|     187k |   11.4 | 1.675918 | --fp8 --attn-impl flex |

Less than 1% toks/sec improvement. It did improve bpb. However, NANOCHAT_FA2_WINDOWED_FLASH=1 improved toks/sec
significantly more while at the same time matching flex attention's bpb improvment.

## License

MIT
