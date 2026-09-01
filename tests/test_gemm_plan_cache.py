"""The NT wgrad plan cache's tier, mode, JSON and locking logic -- all of it without a GPU.

This is why `gemm_cache` is its own module: inside `fp8_pinned` every one of these would sit
behind that file's CUDA-only import guard. The parts that genuinely need a card (restore
fidelity, self-heal end to end, enumeration) live in tests/test_fp8_wgrad_nt.py.
"""
import argparse
import glob
import json
import os

import pytest

from nanochat.sm120 import gemm_cache

CFG = {"algo_id": 21, "tile": 20, "stages": 15, "splitk": 1, "reduction": 0,
       "swizzle": 0, "custom": 0, "inner_shape": 0, "cluster_shape": 0}
INFO = {"us": 47.3, "heuristic_us": 51.9, "waves": 1.0}
KEY = "768x3072x2048|float8_e5m2|float8_e4m3fn|nt|accum0|fa0"
OTHER = "768x768x2048|float8_e5m2|float8_e4m3fn|nt|accum0|fa0"


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """A private base dir and a fixed environment, so nothing here touches the real cache."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(gemm_cache, "_device_slug", lambda: "test-gpu")
    monkeypatch.setattr(gemm_cache, "_cc_slug", lambda: "cc120")
    gemm_cache.reset()
    gemm_cache.set_env(lt_version=130300, ext_tag="test")
    yield
    gemm_cache.reset()


def store(tier_exhaustive=False, max_candidates=512, key=KEY, mode="use"):
    """Write one entry at the requested tier and return the budget it was stored under."""
    gemm_cache.configure(mode, exhaustive=tier_exhaustive, max_candidates=max_candidates)
    b = gemm_cache.budget(64, 20, 8, 5)
    gemm_cache.store(key, dict(CFG), dict(INFO), b)
    return b


def lookup(tier_exhaustive=False, max_candidates=512, key=KEY, mode="use"):
    gemm_cache.configure(mode, exhaustive=tier_exhaustive, max_candidates=max_candidates)
    return gemm_cache.lookup(key, gemm_cache.budget(64, 20, 8, 5))


def read_doc():
    with open(gemm_cache.path()) as f:
        return json.load(f)


# --- the tier rule: the reason this feature is more than a dict on disk ---

def test_heuristic_entry_serves_heuristic_request():
    store(tier_exhaustive=False)
    entry = lookup(tier_exhaustive=False)
    assert entry is not None and entry["tier"] == "heuristic"
    assert entry["algo"] == CFG


def test_heuristic_entry_misses_exhaustive_request():
    """The case the tier exists for: populate with ordinary runs, then ask for a deeper search."""
    store(tier_exhaustive=False)
    assert lookup(tier_exhaustive=True) is None


def test_exhaustive_entry_serves_heuristic_request():
    """Upgrades are monotone -- a stronger entry is still valid for a weaker request."""
    store(tier_exhaustive=True)
    entry = lookup(tier_exhaustive=False)
    assert entry is not None and entry["tier"] == "exhaustive"


def test_exhaustive_entry_replaces_heuristic_entry():
    store(tier_exhaustive=False)
    store(tier_exhaustive=True)
    assert read_doc()["plans"][KEY]["tier"] == "exhaustive"
    assert lookup(tier_exhaustive=True) is not None


def test_budget_is_monotone():
    store(tier_exhaustive=True, max_candidates=512)
    assert lookup(tier_exhaustive=True, max_candidates=128) is not None   # stored >= requested
    assert lookup(tier_exhaustive=True, max_candidates=1024) is None      # stored <  requested


def test_unknown_tier_is_a_miss():
    store()
    doc = read_doc()
    doc["plans"][KEY]["tier"] = "something-else"
    with open(gemm_cache.path(), "w") as f:
        json.dump(doc, f)
    assert lookup() is None


# --- modes ---

def test_mode_off_never_reads_or_writes():
    """The default-behaviour-unchanged guarantee, at the module level."""
    store(mode="use")
    before = open(gemm_cache.path()).read()

    gemm_cache.configure("off")
    assert gemm_cache.lookup(KEY, gemm_cache.budget(64, 20, 8, 5)) is None
    gemm_cache.store(OTHER, dict(CFG), dict(INFO), gemm_cache.budget(64, 20, 8, 5))
    assert open(gemm_cache.path()).read() == before
    assert gemm_cache.log_lines() == []


def test_mode_refresh_ignores_but_overwrites():
    store(mode="use")
    store(key=OTHER, mode="use")

    gemm_cache.configure("refresh")
    assert gemm_cache.lookup(KEY, gemm_cache.budget(64, 20, 8, 5)) is None   # deliberate re-tune
    gemm_cache.store(KEY, {**CFG, "algo_id": 99}, dict(INFO), gemm_cache.budget(64, 20, 8, 5))

    plans = read_doc()["plans"]
    assert plans[KEY]["algo"]["algo_id"] == 99
    assert OTHER in plans, "refresh must not drop entries for other shapes"


def test_absent_key_is_a_miss():
    store()
    assert lookup(key=OTHER) is None


# --- self-healing ---

def test_env_mismatch_discards_and_warns():
    store()
    doc = read_doc()
    doc["env"]["ext_tag"] = "built-against-something-else"
    with open(gemm_cache.path(), "w") as f:
        json.dump(doc, f)

    assert lookup() is None
    assert any("ext_tag" in line for line in gemm_cache.log_lines())


def test_format_bump_discards():
    store()
    doc = read_doc()
    doc["format"] = 999
    with open(gemm_cache.path(), "w") as f:
        json.dump(doc, f)
    assert lookup() is None


def test_corrupt_json_self_heals():
    store()
    with open(gemm_cache.path(), "w") as f:
        f.write('{"format": 1, "plans": {"trunc')

    assert lookup() is None
    assert any("unreadable" in line for line in gemm_cache.log_lines())

    store()                                   # writing over the corruption must succeed
    assert lookup() is not None


def test_missing_file_is_a_quiet_miss():
    gemm_cache.configure("use")
    assert gemm_cache.lookup(KEY, gemm_cache.budget(64, 20, 8, 5)) is None
    assert gemm_cache.log_lines() == []       # absence is normal, not a warning


# --- writing ---

def test_store_merges_and_leaves_no_temp_files():
    store(key=KEY)
    store(key=OTHER)
    plans = read_doc()["plans"]
    assert set(plans) == {KEY, OTHER}, "a second store must not clobber the first"
    assert glob.glob(os.path.join(os.path.dirname(gemm_cache.path()), "*.tmp*")) == []


def test_invalidate_removes_only_its_key():
    store(key=KEY)
    store(key=OTHER)
    gemm_cache.invalidate(KEY, "verify")
    assert set(read_doc()["plans"]) == {OTHER}


def test_non_writer_rank_does_not_write(monkeypatch):
    """Only rank 0 writes: two ranks tuning independently would make the file order-dependent."""
    monkeypatch.setattr(gemm_cache, "_is_writer", lambda: False)
    gemm_cache.configure("use")
    gemm_cache.store(KEY, dict(CFG), dict(INFO), gemm_cache.budget(64, 20, 8, 5))
    assert gemm_cache.path() is not None and not os.path.exists(gemm_cache.path())


def test_stored_entry_shape():
    b = store(tier_exhaustive=True, max_candidates=256)
    entry = read_doc()["plans"][KEY]
    assert entry["tier"] == "exhaustive"
    assert entry["budget"] == b
    assert entry["algo"] == CFG               # ints, not floats -- the file stays diffable
    assert entry["tuned_at"].endswith("Z")


def test_path_is_none_before_set_env(monkeypatch):
    monkeypatch.setattr(gemm_cache, "_LT_VERSION", None)
    assert gemm_cache.path() is None
    gemm_cache.configure("use")
    assert gemm_cache.lookup(KEY, gemm_cache.budget(64, 20, 8, 5)) is None


# --- validation and flag plumbing ---

def test_configure_rejects_bad_values():
    with pytest.raises(ValueError):
        gemm_cache.configure("sometimes")
    with pytest.raises(ValueError):
        gemm_cache.configure("use", max_candidates=0)
    assert gemm_cache.mode() == "off", "a rejected value must not half-configure the module"


def test_plan_key_separates_shapes_and_dtypes():
    import torch

    a = gemm_cache.plan_key(768, 3072, 2048, torch.float8_e5m2, torch.float8_e4m3fn, "nt")
    b = gemm_cache.plan_key(768, 3072, 2048, torch.float8_e4m3fn, torch.float8_e5m2, "nt")
    c = gemm_cache.plan_key(768, 768, 2048, torch.float8_e5m2, torch.float8_e4m3fn, "nt")
    assert len({a, b, c}) == 3
    assert "torch." not in a


def test_flags_default_to_off():
    """recipe.add_args has never had coverage; these defaults are the no-op guarantee."""
    from nanochat.sm120 import recipe

    p = argparse.ArgumentParser()
    recipe.add_args(p)
    args = p.parse_args([])
    assert args.gemm_plan_cache == "off"
    assert args.gemm_autotune_exhaustive is False
    assert args.gemm_autotune_max_candidates == 512


def test_flags_parse_when_passed():
    from nanochat.sm120 import recipe

    p = argparse.ArgumentParser()
    recipe.add_args(p)
    args = p.parse_args(["--gemm-plan-cache", "refresh", "--gemm-autotune-exhaustive",
                         "--gemm-autotune-max-candidates", "128"])
    assert args.gemm_plan_cache == "refresh"
    assert args.gemm_autotune_exhaustive is True
    assert args.gemm_autotune_max_candidates == 128


def test_bad_cache_mode_is_rejected_by_argparse():
    from nanochat.sm120 import recipe

    p = argparse.ArgumentParser()
    recipe.add_args(p)
    with pytest.raises(SystemExit):
        p.parse_args(["--gemm-plan-cache", "maybe"])


def _args(**over):
    """A parsed-args stand-in for recipe.apply's early-return paths."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--fp8", action="store_true")
    from nanochat.sm120 import recipe

    recipe.add_args(p)
    args = p.parse_args([])
    for k, v in over.items():
        setattr(args, k, v)
    return args


def test_gemm_flags_asked():
    from nanochat.sm120 import recipe

    assert not recipe._gemm_flags_asked(_args())
    assert recipe._gemm_flags_asked(_args(gemm_plan_cache="use"))
    assert recipe._gemm_flags_asked(_args(gemm_autotune_exhaustive=True))


def test_cache_flags_without_fp8_warn(capsys):
    """A flag that governs nothing must say so: silence would let someone believe a cache is on."""
    from nanochat.sm120 import recipe

    recipe.apply(None, _args(fp8=False, gemm_plan_cache="use"), "cpu")
    assert "--gemm-plan-cache" in capsys.readouterr().out


def test_no_cache_flags_no_warning(capsys):
    from nanochat.sm120 import recipe

    recipe.apply(None, _args(fp8=False), "cpu")
    assert "gemm" not in capsys.readouterr().out
