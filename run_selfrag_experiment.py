#!/usr/bin/env python3
from __future__ import annotations

import argparse

from selfrag_experiment.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a standalone SELF-RAG experiment with stable artifacts.")
    parser.add_argument("--dataset-name", required=True, help="Dataset name used for the run folder path.")
    parser.add_argument("--default-yaml", required=True, help="Reference YAML file to resolve experiment defaults from.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing run by skipping already written query_ids.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = run_experiment(dataset_name=args.dataset_name, default_yaml_path=args.default_yaml, resume=args.resume)
    print(str(run_root))


if __name__ == "__main__":
    main()

