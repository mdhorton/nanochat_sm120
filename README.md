## NOTE: This is a fork of the nanochat repo: https://github.com/karpathy/nanochat

# Nanochat for sm120 GPUs

This fork explores sm120 GPU peformance with nanochat. Here are a few questions I was curious about:

- How does sm120 training and inference performance compare with H100 (sm90)?
- Can Blackwell specific features (eg, nvfp4) help close the gap?
- How does rental cost compare bwtewen sm120 and sm90?

# TLDR; Conclusion

TBD

## Baseline

Nanochat initially outputs the following when run on a sm120 GPU:

```
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback
WARNING: Training will be less efficient without FA3
WARNING: SDPA has no support for sliding window attention (window_pattern='SSSL'). Your GPU utilization will be terrible.
WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.
```

This is very useful and tells you where to look for performance improvements. Using `--window-pattern L` helps quite a
bit but it's more of a temporary workaround.

Runs using `--window-pattern L` represent the nanochat upstream baseline for sm120.

## What are sm120 GPUs?

These are non-datacenter Blackwell GPUs. For example:

- RTX Pro 6000
- RTX Pro 4000
- RTX 5090

They are significantly cheaper vs datacenter Blackwell. However, sm120 GPUs lack several key performance features:

- no tcgen05
- no tensor core memory
- no NVLink

Furthermore, the non-Pro line (ie, RTX 5090) lack the following:

- no ECC vram
- no P2P capability

These missing features mean that the latest versions of FlashAttention (FA3+) will work on sm120 GPUs.

# Benchmark results

## runs/shortrun.sh (--num-iterations 50)

These test runs usually complete within 5-10 minutes. Full runs are performed later for model validation.

## --depth 24 --device-batch-size 16

Nanochat runs are usually depth=24, which requires ~60 GB VRAM.

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

## --depth 12 --device-batch-size 8

These GPUs have less VRAM so depth and device-batch-size must be reduced. This increases training speed, but at the cost
of model quality.

### training precision: bf16

base flag: none

#### 4x RTX Pro 4000

| toks/sec | mem GB |      bpb | extra flags            |                   
|---------:|-------:|---------:|------------------------|
|     197k |    9.6 | 1.672274 |                        |                   
|     275k |    9.6 | 1.682605 | --window-pattern L     |                   
|     286k |    9.6 | 1.672474 | NANOCHAT_FA2_SWINDOW=1 |      

#### 2x RTX Pro 4000

| toks/sec | mem GB |      bpb | extra flags            |                   
|---------:|-------:|---------:|------------------------|
|     112k |   10.2 | 1.676620 |                        |                   
|     165k |   10.2 | 1.686013 | --window-pattern L     |                   
|     172k |   10.2 | 1.676444 | NANOCHAT_FA2_SWINDOW=1 |                   

### training precision: fp8

base flag: --fp8

#### 4x RTX Pro 4000

| toks/sec | mem GB |      bpb | extra flags                                             |                   
|---------:|-------:|---------:|---------------------------------------------------------|
|     208k |   10.9 | 1.677617 |                                                         |                   
|     302k |   10.8 | 1.687429 | --window-pattern L                                      |  
|     316k |   10.8 | 1.677830 | NANOCHAT_FA2_SWINDOW=1                                  |                   
|     328k |    9.1 | 1.668514 | NANOCHAT_FA2_SWINDOW=1 --fp8-scaling delayed            |                   
|     362k |   10.1 | 1.661808 | NANOCHAT_FA2_SWINDOW=1 --fp8-scaling delayed --wgrad-nt |     

#### 2x RTX Pro 4000

| toks/sec | mem GB |      bpb | extra flags                                             |                   
|---------:|-------:|---------:|---------------------------------------------------------|
|     120k |   11.4 | 1.676669 |                                                         |                   
|     185k |   11.4 | 1.686771 | --window-pattern L                                      |  
|     195k |   11.4 | 1.675813 | NANOCHAT_FA2_SWINDOW=1                                  |                   
|     204k |    9.7 | 1.662816 | NANOCHAT_FA2_SWINDOW=1 --fp8-scaling delayed            |                   
|     229k |   10.6 | 1.662855 | NANOCHAT_FA2_SWINDOW=1 --fp8-scaling delayed --wgrad-nt |                   

#### training precision: nvfp4

base flag: --nvfp4

#### 4x RTX Pro 4000

| toks/sec | mem GB |      bpb | extra flags                                    |
|---------:|-------:|---------:|------------------------------------------------|
|     231k |   14.0 | 1.683095 |                                                |                                          
|     351k |   14.0 | 1.692907 | --window-pattern L                             |                                          
|     367k |   14.0 | 1.682989 | NANOCHAT_FA2_SWINDOW=1                         |  
|     370k |   14.0 | 1.683293 | NANOCHAT_FA2_SWINDOW=1 --nvfp4-scaling delayed |    

#### 2x RTX Pro 4000

| toks/sec | mem GB |      bpb | extra flags                                    |
|---------:|-------:|---------:|------------------------------------------------|
|     135k |   14.0 | 1.681464 |                                                |                                          
|     220k |   14.0 | 1.691275 | --window-pattern L                             |                                          
|     233k |   14.0 | 1.681451 | NANOCHAT_FA2_SWINDOW=1                         |  
|     236k |   14.0 | 1.682219 | NANOCHAT_FA2_SWINDOW=1 --nvfp4-scaling delayed |

## Accepted

### FA2 sliding-window kernels

NANOCHAT_FA2_SWINDOW=1

This addition is mostly just wiring code so that existing FA2 kernels are used for sliding-windows.

FA3 has no sm120 kernels, so the `SSSL` sliding-window default uses an explicit SDPA mask, which is slow. The
`--window-pattern L` option helps but it's more of a workaround and not optimal. This new option allows it to use FA2
kernels for the S layers instead of SDPA masking.

The switch is an environment variable rather than a flag because it has to reach every entry point, including the eval
paths that take no flags. This is an acceptable trade off given that the purpose of this repo is to explore and learn.

## Rejected

### FlexAttention

This was the first attempt to fix the sliding-window problem.

https://pytorch.org/blog/flexattention/

branch: flex-attention

result: Less than 1% toks/sec improvement; also a minor bpb improvement. Overall not exciting. Compare this with
NANOCHAT_FA2_SWINDOW=1, which improved toks/sec significantly more while also improving bpb.

### Exhaustive optimal gemm search

branch: exhaustive-search

result: Increased startup time and code complexity with near zero performance improvement.

## Papers

https://arxiv.org/pdf/2601.22813

https://arxiv.org/pdf/2605.28213

## License

MIT
