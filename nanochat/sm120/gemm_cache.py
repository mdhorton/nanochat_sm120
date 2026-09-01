"""Persistent cuBLASLt plan cache for the NT wgrad autotuner (`--gemm-plan-cache`).

`fp8_pinned` autotunes a plan per shape at first use and keeps it in a process-local dict, so
every run re-tunes from scratch. Selection is by measured microseconds, so two algorithms within
timing noise swap winners between otherwise identical runs -- which makes a benchmark arm
non-reproducible at the kernel level for reasons that have nothing to do with the model.

This persists the winner's *config attributes* (algo id, tile, stages, splitK, reduction,
swizzle, custom option, inner and cluster shape) rather than the opaque `cublasLtMatmulAlgo_t`
itself: that struct is a POD whose layout carries no cross-version guarantee, and memcpy'ing it
between cuBLAS releases would restore silent garbage. `PinnedGemm.restore` rebuilds from the
attributes through `AlgoInit` + `ConfigSetAttribute` + `AlgoCheck`.

Each entry records *which search found it*. A run asking for a stronger search than the entry
was tuned with is a miss, so populating the cache with ordinary runs and adding
`--gemm-autotune-exhaustive` later re-tunes instead of silently replaying the weaker pick.
Upgrades are monotone: an exhaustive entry serves a heuristic-tier run, never the reverse.

The GPU and the cuBLASLt version are in the *filename*, not just the body, following the build
directory at `fp8_pinned._ext()`. Algo ids are not stable across cuBLAS releases, and a bump
that cannot even name the old file is a stronger guarantee than one that has to detect it.

This is a separate module from `fp8_pinned` so the tier, JSON and locking logic can be tested
without a GPU; inside `fp8_pinned` it would sit behind that module's CUDA-only import guard.
"""
import json
import os
import re
import sys
import time

import torch

_MODES = ("off", "use", "refresh")
_FORMAT = 1
# Ordering is the whole point: a request may be served by an entry of equal or higher rank only.
_TIER_RANK = {"heuristic": 0, "exhaustive": 1}

_MODE = "off"
_EXHAUSTIVE = False
_MAX_CANDIDATES = 512
_LT_VERSION = None
_EXT_TAG = None
_LOG = []
_ENV_NOTED = False


def configure(mode, exhaustive=False, max_candidates=512):
    """Set the cache mode. Called once from the recipe before training starts.

    Validates before assigning anything, so a bad value cannot leave the module half-configured.
    """
    global _MODE, _EXHAUSTIVE, _MAX_CANDIDATES
    if mode not in _MODES:
        raise ValueError(f"--gemm-plan-cache must be one of {_MODES}, got {mode!r}")
    max_candidates = int(max_candidates)
    if max_candidates < 1:
        raise ValueError(f"--gemm-autotune-max-candidates must be >= 1, got {max_candidates}")
    _MODE, _EXHAUSTIVE, _MAX_CANDIDATES = mode, bool(exhaustive), max_candidates


def set_env(lt_version, ext_tag):
    """Record the loaded cuBLASLt version and extension tag. Called by fp8_pinned._ext().

    The version has to come from the extension -- it links the toolkit's libcublasLt, not
    torch's -- so it arrives by injection rather than by importing fp8_pinned back.
    """
    global _LT_VERSION, _EXT_TAG
    _LT_VERSION, _EXT_TAG = int(lt_version), str(ext_tag)


def mode():
    return _MODE


def enabled():
    return _MODE != "off"


def exhaustive():
    return _EXHAUSTIVE


def max_candidates():
    return _MAX_CANDIDATES


def tier():
    return "exhaustive" if _EXHAUSTIVE else "heuristic"


def budget(max_algos, iters, finalists, rounds):
    """The measurement budget an entry was tuned under. Compared component-wise on lookup."""
    return {"max_algos": int(max_algos), "iters": int(iters), "finalists": int(finalists),
            "rounds": int(rounds),
            "max_candidates": int(_MAX_CANDIDATES) if _EXHAUSTIVE else 0}


def log_lines():
    """Drained by recipe.PerfStack.report_once, via fp8_pinned.log_lines()."""
    return list(_LOG)


def reset():
    """Test hook: forget mode and accumulated log lines."""
    global _MODE, _EXHAUSTIVE, _MAX_CANDIDATES, _ENV_NOTED
    _MODE, _EXHAUSTIVE, _MAX_CANDIDATES, _ENV_NOTED = "off", False, 512, False
    _LOG.clear()


def _warn(line):
    """Surface a cache problem where it happens as well as into _LOG.

    Mirrors fp8_pinned._warn: _LOG is only drained after the first backward, and a harness that
    never gets there would otherwise lose the message entirely.
    """
    _LOG.append(line)
    if _is_writer():
        print(f"WARNING: gemm plan cache {line.strip()}", file=sys.stderr, flush=True)


def _is_writer():
    """Rank 0, or a single-process run.

    Only one rank writes: two ranks tuning the same shapes on separate GPUs make two independent
    picks, and last-writer-wins would leave the file's contents dependent on scheduling -- the
    exact non-determinism this cache exists to remove. Other ranks tune and discard, then restore
    rank 0's plan from the next run onward.
    """
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _device_slug():
    try:
        return re.sub(r"[^a-z0-9]+", "-", torch.cuda.get_device_name(0).lower()).strip("-")
    except Exception:
        return "unknown"


def _cc_slug():
    try:
        major, minor = torch.cuda.get_device_capability(0)
        return f"cc{major}{minor}"
    except Exception:
        return "ccunknown"


def plan_key(m, n, k, a_dtype, b_dtype, kind, accum=False, fast_accum=False):
    """The per-plan cache key, as a string because JSON object keys must be strings.

    accum/fast_accum are always (False, False) for the NT wgrad but are in the key anyway, so the
    file survives the --pin-gemm surface that shares this extension being ported.
    """
    a = str(a_dtype).replace("torch.", "")
    b = str(b_dtype).replace("torch.", "")
    return f"{m}x{n}x{k}|{a}|{b}|{kind}|accum{int(accum)}|fa{int(fast_accum)}"


def path():
    """The cache file for this GPU and cuBLASLt version, or None before set_env()."""
    if _LT_VERSION is None:
        return None
    from nanochat.common import get_base_dir

    name = f"wgradnt-{_device_slug()}-{_cc_slug()}-lt{_LT_VERSION}.json"
    return os.path.join(get_base_dir(), "gemm_plans", name)


def _lock(file_path):
    from filelock import FileLock

    # The write happens inside the first backward with the other rank waiting at a collective, so
    # a lock that never resolves would trip the NCCL watchdog rather than merely stalling.
    return FileLock(file_path + ".lock", timeout=10)


def _env():
    return {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
            "cc": _cc_slug(), "cublaslt": _LT_VERSION, "ext_tag": _EXT_TAG,
            "workspace_bytes": 32 * 1024 * 1024}


def _blank():
    return {"format": _FORMAT, "env": _env(), "plans": {}}


def _read_unlocked(file_path):
    """Parse the cache file, or return a blank one. Corruption self-heals rather than raising."""
    try:
        with open(file_path) as f:
            doc = json.load(f)
    except FileNotFoundError:
        return _blank(), False
    except Exception as e:
        _warn(f"file unreadable ({type(e).__name__}: {e}), starting fresh")
        return _blank(), False

    if doc.get("format") != _FORMAT:
        _warn(f"format {doc.get('format')} != {_FORMAT}, discarding")
        return _blank(), False
    # Redundant with the filename on purpose: catches a file copied between machines, or an
    # extension rebuilt against a different toolkit without the cuBLASLt version moving.
    have, want = doc.get("env", {}), _env()
    for field in ("cublaslt", "ext_tag"):
        if have.get(field) != want[field]:
            _warn(f"{field} changed ({have.get(field)} -> {want[field]}), discarding")
            return _blank(), False
    doc.setdefault("plans", {})
    return doc, True


def _note_env_once(file_path, existed):
    """Say so when this environment has no cache but a sibling does -- i.e. a version bump."""
    global _ENV_NOTED
    if _ENV_NOTED or existed:
        return
    _ENV_NOTED = True
    siblings = []
    try:
        d = os.path.dirname(file_path)
        siblings = [f for f in os.listdir(d)
                    if f.startswith("wgradnt-") and f.endswith(".json")
                    and f != os.path.basename(file_path)]
    except OSError:
        pass
    if siblings:
        _LOG.append(f"gemm plan cache: no entries for cublasLt {_LT_VERSION} on this GPU "
                    f"({len(siblings)} cache(s) for another environment) -- tuning fresh")


def _acceptable(entry, want_tier, want_budget):
    """Whether a stored entry may serve this request. All tier logic lives here."""
    have_tier = entry.get("tier")
    if have_tier not in _TIER_RANK:
        return False
    if _TIER_RANK[have_tier] < _TIER_RANK[want_tier]:
        return False                      # stronger search requested: miss, re-tune, overwrite
    if have_tier != want_tier:
        return True                       # strictly higher tier serves a lower request
    have_budget = entry.get("budget", {})
    return all(have_budget.get(k, -1) >= v for k, v in want_budget.items())


def lookup(key, want_budget):
    """The stored entry for this shape, or None on any kind of miss."""
    if _MODE != "use":
        return None                       # 'off' never reads; 'refresh' deliberately re-tunes
    file_path = path()
    if file_path is None:
        return None
    try:
        with _lock(file_path):
            doc, existed = _read_unlocked(file_path)
    except Exception as e:
        _warn(f"could not read {file_path} ({type(e).__name__}: {e})")
        return None
    _note_env_once(file_path, existed and bool(doc["plans"]))
    entry = doc["plans"].get(key)
    if entry is None:
        return None
    if not _acceptable(entry, tier(), want_budget):
        return None
    return entry


def store(key, cfg, info, want_budget):
    """Persist a freshly tuned plan. Never raises -- a cache failure must not end a training run."""
    if not enabled() or not _is_writer():
        return
    file_path = path()
    if file_path is None:
        return
    entry = {
        "tier": tier(),
        "budget": dict(want_budget),
        "algo": {k: int(v) for k, v in sorted(cfg.items())},
        "us": round(float(info.get("us", 0.0)), 3),
        "heuristic_us": round(float(info.get("heuristic_us", 0.0)), 3),
        "waves": float(info.get("waves", 0.0)),
        "tuned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _mutate(file_path, lambda plans: plans.__setitem__(key, entry))


def invalidate(key, reason):
    """Drop a stale entry so the re-tune that follows can replace it."""
    if not enabled() or not _is_writer():
        return
    file_path = path()
    if file_path is None:
        return
    _mutate(file_path, lambda plans: plans.pop(key, None))


def _mutate(file_path, fn):
    """Read-merge-replace under the lock, so a concurrent writer's entries are never lost."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with _lock(file_path):
            doc, _ = _read_unlocked(file_path)
            fn(doc["plans"])
            doc["format"], doc["env"] = _FORMAT, _env()
            tmp = f"{file_path}.tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(doc, f, indent=1, sort_keys=True)
            os.replace(tmp, file_path)    # atomic: a reader never sees a torn file
    except Exception as e:
        _warn(f"could not write {file_path} ({type(e).__name__}: {e})")
