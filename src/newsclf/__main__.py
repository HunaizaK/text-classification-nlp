"""CLI: python -m newsclf {all,baseline,bert,compress,distill,report} [--full]"""

from __future__ import annotations

import argparse
import logging
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from newsclf.config import load_config  # noqa: E402
from newsclf.pipeline import STEPS, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(prog="newsclf")
    parser.add_argument("step", choices=["all", *STEPS])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--full", action="store_true", help="larger subsets (much slower on CPU)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(args.config, full=args.full)
    run(cfg, list(STEPS) if args.step == "all" else [args.step])


if __name__ == "__main__":
    main()
