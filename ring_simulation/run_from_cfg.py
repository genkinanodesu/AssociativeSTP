"""
Run ring STP demos for configs specified in JSON files.

JSON format:
- Either flat dict with RingConfig overrides, or
- {"cfg_name": "...", "ring": {...}, "replay": {...}}
Missing keys fall back to RingConfig/ReplayConfig defaults.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, asdict
from pathlib import Path
from typing import Any, Dict, Tuple, List

from ring_stp_exactgrad import RingConfig, ReplayConfig
from ring_stp_plotting import run_example_with_cfg, run_replay_demo_with_cfg


STP_TYPES = ("with_stp", "pre_stp", "no_stp")


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to read JSON at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"JSON must be an object at top level: {path}")
    return data


def _split_overrides(data: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    cfg_name = str(data.get("cfg_name")) if "cfg_name" in data else ""
    if "ring" in data or "replay" in data:
        ring_overrides = data.get("ring", {})
        replay_overrides = data.get("replay", {})
        if ring_overrides is None:
            ring_overrides = {}
        if replay_overrides is None:
            replay_overrides = {}
    else:
        ring_overrides = {k: v for k, v in data.items() if k != "cfg_name"}
        replay_overrides = {}
    if not isinstance(ring_overrides, dict) or not isinstance(replay_overrides, dict):
        raise ValueError("ring/replay sections must be objects")
    return cfg_name, ring_overrides, replay_overrides


def _apply_overrides(cfg: Any, overrides: Dict[str, Any], label: str) -> None:
    valid = {f.name for f in fields(type(cfg))}
    unknown = []
    for key, val in overrides.items():
        if key in valid:
            setattr(cfg, key, val)
        else:
            unknown.append(key)
    if unknown:
        print(f"[{label}] Ignoring unknown keys: {', '.join(unknown)}")


def _stp_overrides(stp_type: str) -> Dict[str, Any]:
    if stp_type == "with_stp":
        return {"stp_enabled": True}
    if stp_type == "pre_stp":
        return {"stp_enabled": True, "U_init_jitter": 0.0, "lr_U": 0.0}
    if stp_type == "no_stp":
        return {"stp_enabled": False, "U_init_jitter": 0.0}
    raise ValueError(f"Unknown stp_type: {stp_type}")


def _unique_ordered(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for val in values:
        if val not in seen:
            out.append(val)
            seen.add(val)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ring STP demos from JSON configs.")
    parser.add_argument("cfg_paths", nargs="+", help="JSON config paths.")
    parser.add_argument(
        "--stp-type",
        action="append",
        choices=STP_TYPES,
        help="STP condition to run; can be repeated (default: run all).",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "example", "replay"],
        default="all",
        help="Select which demo to run (default: all).",
    )
    parser.add_argument("--force-retrain", action="store_true", help="Ignore cached fits and retrain.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (blocks).",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Save paper-ready figures under saved_runs/<cfg>/paper and adjust plotting.",
    )
    args = parser.parse_args()

    stp_types = _unique_ordered(args.stp_type) if args.stp_type else list(STP_TYPES)

    for cfg_path in args.cfg_paths:
        path = Path(cfg_path)
        data = _load_json(path)
        cfg_name, ring_overrides, replay_overrides = _split_overrides(data)
        if not cfg_name:
            cfg_name = path.stem

        base_cfg = RingConfig()
        _apply_overrides(base_cfg, ring_overrides, label=str(path))

        rp = None
        if replay_overrides:
            rp = ReplayConfig()
            _apply_overrides(rp, replay_overrides, label=str(path))

        for stp_type in stp_types:
            cfg = RingConfig(**asdict(base_cfg))
            _apply_overrides(cfg, _stp_overrides(stp_type), label=f"{path} ({stp_type})")
            run_cfg_name = str(Path(cfg_name) / stp_type)

            if args.mode in ("all", "example"):
                run_example_with_cfg(
                    cfg,
                    cfg_name=run_cfg_name,
                    force_retrain=args.force_retrain,
                    show=args.show,
                    paper=args.paper,
                )
            if args.mode in ("all", "replay"):
                replay_force_retrain = args.force_retrain if args.mode != "all" else False
                run_replay_demo_with_cfg(
                    cfg,
                    rp=rp,
                    cfg_name=run_cfg_name,
                    force_retrain=replay_force_retrain,
                    show=args.show,
                    paper=args.paper,
                )


if __name__ == "__main__":
    main()
