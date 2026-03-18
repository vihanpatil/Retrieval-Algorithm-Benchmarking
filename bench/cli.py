from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .datasets import load_bundle_from_config
from .report import build_report
from .runner import run_benchmark


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    bundle = load_bundle_from_config(cfg)
    out_jsonl = cfg.get("output", {}).get("results_jsonl", "artifacts/results/results.jsonl")
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    run_benchmark(cfg, bundle, out_jsonl)


def cmd_report(args: argparse.Namespace) -> None:
    build_report(args.results, args.outdir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dense ANN benchmark harness")
    sub = p.add_subparsers(required=True)

    p_run = sub.add_parser("run", help="Run benchmark sweeps")
    p_run.add_argument("--config", required=True, help="Path to YAML config")
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="Build summary CSV and plots")
    p_report.add_argument("--results", required=True, help="Path to JSONL results")
    p_report.add_argument("--outdir", required=True, help="Output folder for reports")
    p_report.set_defaults(func=cmd_report)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

