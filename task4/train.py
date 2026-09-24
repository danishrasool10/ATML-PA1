"""Train a method from a YAML config."""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    method = cfg["method"].lower()
    if method == "vanilla":
        from methods.vanilla import run
    elif method == "gcsc":
        from methods.gcsc import run
    elif method == "proser":
        from methods.proser import run
    else:
        raise ValueError(f"Unknown method: {method}")
    run(cfg)


if __name__ == "__main__":
    main()