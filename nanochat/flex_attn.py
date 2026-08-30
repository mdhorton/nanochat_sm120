"""
FlexAttention training path for sm120 (consumer/workstation Blackwell), where FA3 is unavailable.

Why: without FA3, `nanochat/flash_attention.py` falls back to SDPA, and a sliding window there means
building a dense TxT bool mask, which drops SDPA onto the mem-efficient backend. Measured on an
RTX PRO 4000 (B=8 H=10 T=2048 D=128, bf16, fwd+bwd, dev/bench_attention.py):

    SDPA causal                     3.82 ms   <- L layers today
    SDPA + dense mask, window=512  23.62 ms   <- S layers today (6x slower than no window at all)
    flex, causal BlockMask          5.74 ms
    flex, sliding BlockMask (512)   2.97 ms

So flex wins big on windowed layers and *loses* on dense causal. Layers whose window covers the whole
sequence therefore keep using the existing FA3/SDPA path (see BlockMaskCache.build returning None).

Training only: flex_attention must run inside torch.compile. In eager it materializes the score matrix,
which would be catastrophic for the uncompiled inference/eval paths (Engine, generate, evaluate_core).
`CausalSelfAttention.forward` guards on `kv_cache is None` for exactly this reason.
"""
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks


def flex_attn_func(q, k, v, block_mask):
    """
    FlexAttention with the same shape contract as flash_attn_func: (B, T, H, D) in and out.

    Masking (causal, sliding window) lives entirely in block_mask, so there is no `causal` or
    `window_size` argument here.
    """
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # -> (B, H, T, D)
    enable_gqa = q.size(1) != k.size(1)
    y = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def _causal_mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _sliding_mask_mod(window):
    """Matches FA3's window_size=(window, 0): attend to keys in [q_idx - window, q_idx]."""
    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx - kv_idx <= window
    return mask_mod


class BlockMaskCache:
    """
    Builds one BlockMask per distinct window size and hands out a per-layer list.

    Sliding-window masks depend only on (window, seq_len), not on the batch, so they are built once
    and reused for the whole run. create_block_mask still goes through torch.compile because eager
    costs 7.24 ms/call vs 0.19 ms compiled on sm120 (matters once masks become batch-dependent).
    """

    def __init__(self, window_sizes, seq_len, device):
        self.window_sizes = list(window_sizes)
        self.seq_len = seq_len
        self.device = device
        self._create = torch.compile(create_block_mask, dynamic=False)
        # Only layers with a real sliding window get a BlockMask; full-context layers stay on SDPA/FA3.
        self._windows = sorted({w for w, _ in self.window_sizes if 0 <= w < seq_len})
        self._masks = None

    def num_masked_layers(self):
        return sum(1 for w, _ in self.window_sizes if 0 <= w < self.seq_len)

    def build(self):
        """Returns list[BlockMask | None] of length n_layer. None means: use the FA3/SDPA path."""
        if self._masks is None:
            by_window = {
                window: self._create(
                    and_masks(_causal_mask_mod, _sliding_mask_mod(window)),
                    None, None, self.seq_len, self.seq_len, device=self.device,
                )
                for window in self._windows
            }
            self._masks = [by_window.get(w) for w, _ in self.window_sizes]
        return self._masks


class FlexEvalWrapper:
    """
    Shim so evaluate_bpb() runs under the same masking as training without touching
    nanochat/loss_eval.py, which only needs get_device() and a callable.
    """

    def __init__(self, model, cache):
        self.model = model
        self.cache = cache

    def get_device(self):
        return self.model.get_device()

    def __call__(self, idx, targets=None, **kwargs):
        return self.model(idx, targets, block_masks=self.cache.build(), **kwargs)


def make_block_mask_cache(attn_impl, model, device, seq_len=None):
    """
    Returns a BlockMaskCache for --attn-impl flex, or None to keep the existing FA3/SDPA path.
    `model` is the uncompiled GPT (for window_sizes and config). `seq_len` overrides
    model.config.sequence_len when training T differs from it (chat_sft --max-seq-len).
    """
    if attn_impl != "flex":
        return None
    assert torch.cuda.is_available() and str(device).startswith("cuda"), \
        "--attn-impl flex requires CUDA (flex_attention has no usable CPU/MPS training path)"
    seq_len = model.config.sequence_len if seq_len is None else seq_len
    return BlockMaskCache(model.window_sizes, seq_len, device)
