"""One place that owns the sm120 flags, the enablement order and the per-step hooks.

scripts/base_train.py keeps four calls instead of the flag plumbing:

    recipe.add_args(parser)                        # before parse_args
    perf = recipe.apply(model, args, device_type)  # before torch.compile
    perf.after_backward()                          # between the last backward and the step
    perf.report_once()                             # after that, once -- what got pinned

The ordering inside a step is load-bearing and documented on the method that owns it: the amax
update has to see this step's readings before the weights move. Getting it wrong is silent --
the loss still falls, just onto stale scales -- which is why it lives here rather than at a
hand-placed site.

--fp8-scaling, --nvfp4-scaling, --pin-gemm and --wgrad-nt are ported; see TODO.md for the rest
of the stack this file is the landing surface for. The NVFP4 history attaches in base_train's
nvfp4 block rather than here, because apply() runs before the NVFP4Linear conversion exists to
attach to.
"""
import torch.nn as nn

from nanochat.common import is_ddp_initialized, print0


def add_args(parser):
    """The sm120 flags. Every one is opt-in; a run that passes none gets upstream behaviour."""
    g = parser.add_argument_group("sm120 performance stack")
    g.add_argument("--fp8-scaling", type=str, default="dynamic",
                   choices=["dynamic", "delayed", "static-spike"],
                   help="fp8 scale source. 'delayed' scales from an amax history instead of the current tensor, so the cast is not gated by a reduction and each quantized tensor is read once instead of twice: +10.7%% at d12. 'static-spike' replaces the amax with a fixed constant: MEASUREMENT ONLY, the gradients are wrong, it exists to price the ceiling that 'delayed' chases")
    g.add_argument("--amax-history", type=int, default=16,
                   help="--fp8-scaling / --nvfp4-scaling delayed: how many past steps the scale is the max over. Longer is more robust to a spike, slower to follow a trend")
    g.add_argument("--amax-margin", type=float, default=2.0,
                   help="--fp8-scaling / --nvfp4-scaling delayed: headroom divisor on the scale, i.e. how far the amax may grow in one step before the cast clips. Nearly free in a floating-point format -- it costs range at the bottom, not mantissa bits; 1.0 leaves no room at all")
    g.add_argument("--amax-allreduce", action="store_true",
                   help="--fp8-scaling / --nvfp4-scaling delayed: all-reduce the amaxes across ranks so every rank picks the same scale. Off by default -- the scale is exactly inverted (by _scaled_mm for fp8, by the block scales for NVFP4), so per-rank divergence changes only rounding")
    g.add_argument("--nvfp4-scaling", type=str, default="dynamic", choices=["dynamic", "delayed"],
                   help="NVFP4 activation scale source. 'delayed' takes the per-tensor amax from a history instead of a vector_norm pre-pass over the activation, and reads the next one back off the e4m3 block scales -- 32x fewer bytes (dev/nvfp4-quartet.md, queue B1). Needs --nvfp4 and 4/6 rounding. Numerics-affecting, so it is not part of the --nvfp4 bundle")
    g.add_argument("--pin-gemm", type=str, default="off",
                   choices=["off", "attn", "wgrad", "all"],
                   help="substitute an autotuned cuBLASLt algorithm for the heuristic's pick on the fp8 GEMMs -- cuBLAS mispicks the wgrad shapes by 15-42%% and fwd/dgrad by 5-25%%, and _scaled_mm cannot ask for another. 'attn' pins only the square attention wgrad, 'wgrad' every wgrad shape, 'all' adds fwd and dgrad: +6.0%% at d12, +0.8%% at d16, the win collapsing with depth. Needs --fp8, and JIT-builds csrc/pinned_gemm.cu on first use")
    g.add_argument("--wgrad-nt", action="store_true",
                   help="run the weight-grad GEMMs in the natural (NT) operand layout, which sm120's cuBLASLt accepts, instead of building the transposed copies the TN form needs -- deletes the pure-copy fp8 transpose kernels, 4.6%% of a step: +8.4%% at d12. Needs --fp8, and JIT-builds csrc/pinned_gemm.cu on first use. Costs ~1.7 GB of peak memory")
    g.add_argument("--fp8-exclude", type=str, default="",
                   help="comma-separated Linear names to keep in bf16 under --fp8, matched against the last component of the module fqn (e.g. 'lm_head'). Each fp8 Linear costs an amax+cast+transpose pass over its activations; for a layer whose GEMM saving is small relative to that traffic, bf16 can win")
    return g


def module_filter(args):
    """The fp8 conversion filter: hardware limits plus whatever --fp8-exclude names."""
    excluded = {n.strip() for n in args.fp8_exclude.split(",") if n.strip()}

    def keep(mod, fqn):
        if not isinstance(mod, nn.Linear):
            return False
        if fqn.split(".")[-1] in excluded:              # --fp8-exclude
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False                                # fp8 hardware requirement
        if min(mod.in_features, mod.out_features) < 128:
            return False                                # too small to pay for the casts
        return True

    return keep, excluded


def apply(model, args, device_type):
    """Convert the model to fp8 and attach whatever the flags asked for. Before torch.compile.

    Returns a PerfStack even when nothing is enabled, so the caller's step loop is
    unconditional. Everything here registers buffers the compiled graph reads, which is why it
    cannot be deferred past the torch.compile call.
    """
    import nanochat.sm120 as sm120
    from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
    from nanochat.sm120 import fp8_pinned, fp8_state

    stack = PerfStack()

    if not args.fp8:
        if args.fp8_scaling != "dynamic":
            print0(f"Warning: --fp8-scaling {args.fp8_scaling} needs --fp8, ignoring")
        if args.wgrad_nt:
            print0("Warning: --wgrad-nt needs --fp8, ignoring")
        if args.pin_gemm != "off":
            print0(f"Warning: --pin-gemm {args.pin_gemm} needs --fp8, ignoring")
        return stack
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
        return stack

    # Must precede the conversion: Float8Linear.__init__ asks the backend to register its
    # buffers, so a backend installed afterwards never sees the layers.
    sm120.install_fp8_backend()
    if args.fp8_scaling == "static-spike":
        fp8_state.set_static_spike(True)
        print0("!! --fp8-scaling static-spike: gradients are WRONG, throughput measurement only")

    keep, excluded = module_filter(args)
    num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    convert_to_float8_training(model, config=Float8LinearConfig.from_recipe_name(args.fp8_recipe),
                               module_filter_fn=keep)
    num_fp8 = sum(1 for m in model.modules() if "Float8" in type(m).__name__)
    excl_note = f", excluded {sorted(excluded)}" if excluded else ""
    print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} "
           f"linear layers, skipped {num_linear - num_fp8} (too small){excl_note}")

    if args.fp8_scaling == "delayed":
        stack.scales = fp8_state.enable_delayed_scaling(
            model,
            history_len=args.amax_history,
            margin=args.amax_margin,
            allreduce=args.amax_allreduce and is_ddp_initialized(),
        )
        ar = ", amax all-reduced" if stack.scales is not None and stack.scales.allreduce else ""
        print0(f"✓ delayed fp8 scaling: {args.amax_history}-step amax history, "
               f"margin {args.amax_margin}{ar}")
    if args.pin_gemm != "off":
        fp8_pinned.configure(args.pin_gemm)
        print0(f"✓ pinned cuBLASLt algorithms: --pin-gemm {args.pin_gemm}, autotuned per shape "
               "on first use")
    if args.wgrad_nt:
        fp8_pinned.configure_wgrad_nt(True)
        print0("✓ natural-layout (NT) wgrad: transpose copies removed from the backward")
    return stack


class PerfStack:
    """The per-step half of the recipe: what has to happen, and in what order, around a step.

    Inert when nothing is enabled -- every hook is a `is not None` test on a handle that stays
    None, so an upstream-shaped run pays one predictable branch per step and nothing else.
    """

    def __init__(self):
        self.scales = None       # DelayedScaleState      (--fp8-scaling delayed)
        self._reported = False

    def after_backward(self):
        """Between the last .backward() and optimizer.step().

        update() has to see this step's amaxes before the weights move: the readings it folds
        into the history were taken against the weights the step is about to replace.
        """
        if self.scales is not None:
            self.scales.update()

    def report_once(self):
        """Plans autotune lazily on first use, so report them after the first backward."""
        from nanochat.sm120 import fp8_pinned

        # The enabled state, not the flag: --pin-gemm/--wgrad-nt without --fp8 have already been
        # warned about and ignored, and warning again here would call a run no-one claimed was a
        # pinned measurement a failed one.
        if self._reported or not fp8_pinned.enabled():
            return
        lines = fp8_pinned.log_lines()
        for line in lines:
            print0(line)
        if not lines:
            # The flag was passed and the first backward is done, so plans should exist. An arm
            # that pinned nothing reads ~2.5% low while still printing its enablement line
            # above, so say so rather than leave the ✓ standing alone.
            print0("WARNING: --pin-gemm/--wgrad-nt built no plans -- this run is NOT using a "
                   "pinned GEMM, and its throughput is not a measurement of one")
        self._reported = True
