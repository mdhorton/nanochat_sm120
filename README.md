## NOTE: This is a fork of the nanochat repo: https://github.com/karpathy/nanochat

# Nanochat for sm120 GPUs

This fork explores sm120 GPU peformance with nanochat. Here are a few questions I was curious about:

1. How does sm120 training throughput compare with H100 (sm90)?
2. Can Blackwell specific features (eg, nvfp4) help close the gap?
3. How does rental cost compare between sm120 and sm90?

# TL;DR conclusion

1. TBD
2. nvfp4 speeds training ~10%, but the gain doesn't justify the hit to model quality. NVIDIA reports success with nvfp4
   training. However, they trained 10 trillion tokens and had significantly better architecture. They were able to keep
   quality in line with fp8. The effort required was justified by the large token count. A sm120 GPU would never see
   this token count. The conclusion I draw is that sm120 nvfp4 is for inference, not training.
3. TBD

## Baseline

Nanochat initially prints the following when run on a sm120 GPU:

```
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback
WARNING: Training will be less efficient without FA3
WARNING: SDPA has no support for sliding window attention (window_pattern='SSSL'). Your GPU utilization will be terrible.
WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.
```

This basically pinpoints where to look for performance improvements. Using `--window-pattern L` improves training speed
quite a bit, but it's more of a workaround.

Benchmarks using `--window-pattern L` are the nanochat upstream baseline for sm120.

## What are sm120 GPUs?

These are non-datacenter Blackwell GPUs. For example:

- RTX Pro 6000
- RTX Pro 4000
- RTX 5090

They are significantly cheaper than datacenter Blackwells. However, sm120 GPUs lack several key performance features:

- no tcgen05
- no tensor core memory
- no thread block clusters
- no NVLink

Furthermore, the non-Pro line (ie, RTX 5090) GPUs lack the following:

- no ECC vram
- no P2P capability

These missing features mean that the latest versions of FlashAttention (FA3+) will not work on sm120 GPUs.

# Benchmark results

## runs/shortrun.sh (--num-iterations 50)

These are short runs to get an idea of throughput. Most complete within 5 minutes.

## --depth 24 --device-batch-size 16

### 8x RTX Pro 6000 WS

| toks/sec | mem GB |      bpb | notes                                                         |
|---------:|-------:|---------:|---------------------------------------------------------------|
|     247k |   60.2 | 1.587426 |                                                               |
|     278k |   52.7 | 1.575156 | --fp8                                                         |
|     350k |   60.2 | 1.602325 | --window-pattern L                                            |
|     417k |   52.7 | 1.589898 | --fp8 --window-pattern L                                      |
|     435k |   52.7 | 1.575121 | --fp8 NANOCHAT_FA2_SWINDOW=1                                  |
|     456k |   60.3 | 1.575138 | --fp8 NANOCHAT_FA2_SWINDOW=1 --wgrad-nt                       |
|     482k |   58.5 | 1.588545 | --fp8 NANOCHAT_FA2_SWINDOW=1 --wgrad-nt --fp8-scaling delayed |
|     585k |   56.2 | 1.582146 | --nvfp4 NANOCHAT_FA2_SWINDOW=1                                |
|     583k |   56.2 | 1.580403 | --nvfp4 NANOCHAT_FA2_SWINDOW=1 --nvfp4-scaling delayed        |

### 2x RTX Pro 6000 WS

| toks/sec | mem GB |      bpb | notes                                                         |
|---------:|-------:|---------:|---------------------------------------------------------------|
|      71k |   64.8 | 1.576989 |                                                               |
|      79k |   57.2 | 1.566846 | --fp8                                                         |
|     107k |   64.8 | 1.590402 | --window-pattern L                                            |
|     118k |   57.2 | 1.582185 | --fp8 --window-pattern L                                      |
|     124k |   57.2 | 1.567681 | --fp8 NANOCHAT_FA2_SWINDOW=1                                  |
|     126k |   55.4 | 1.586394 | --fp8 NANOCHAT_FA2_SWINDOW=1 --fp8-scaling delayed            |
|     130k |   64.8 | 1.566533 | --fp8 NANOCHAT_FA2_SWINDOW=1 --wgrad-nt                       |
|     138k |   63.0 | 1.585184 | --fp8 NANOCHAT_FA2_SWINDOW=1 --fp8-scaling delayed --wgrad-nt |
|     169k |   60.7 | 1.581607 | --nvfp4 NANOCHAT_FA2_SWINDOW=1                                |
|     169k |   60.7 | 1.581782 | --nvfp4 NANOCHAT_FA2_SWINDOW=1 --nvfp4-scaling delayed        |

## --depth 12 --device-batch-size 8

The following benchmarks use GPUs with less VRAM, so depth and device-batch-size must be reduced. This increases
training speed but reduces model quality.

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

This addition is mostly wiring added so that existing FA2 kernels are used for sliding-windows.

FA3 has no sm120 kernels, so the `SSSL` sliding-window default uses an explicit SDPA mask, which is slow. The
`--window-pattern L` option helps but it's limited. This new option allows it to use FA2 kernels for the S layers
instead of SDPA masking.

The option is an environment variable rather than a flag because it has to reach every entry point, including the eval
paths that take no flags. This is an acceptable trade off given the purpose of this repo is to explore and learn.

## Rejected

### NVFP4 training

The goal was to use nvfp4 precision to boost training speed.

https://github.com/IST-DASLab/Quartet-II

result: nvfp4 increased training speed by ~10% vs fp8. Not a bad result, but it introduced quantization noise. fp8
reached the same level of model quality faster than nvfp4 despite nvfp4's faster speed. nvfp4's 10% speed gain isn't
enough to justfiy its use.

### FlexAttention

This was the first attempt to fix the sliding-window problem.

https://pytorch.org/blog/flexattention/

branch: flex-attention

result: Less than 1% toks/sec improvement with a minor bpb improvement. Overall not very significant. Compare this with
NANOCHAT_FA2_SWINDOW=1, which improves toks/sec significantly while also improving bpb.

### Exhaustive optimal gemm search

branch: exhaustive-search

result: Increased startup time and code complexity with near zero performance improvement.

## Papers

https://arxiv.org/pdf/2601.22813

https://arxiv.org/pdf/2605.28213

https://arxiv.org/abs/2509.25149

## License

MIT
