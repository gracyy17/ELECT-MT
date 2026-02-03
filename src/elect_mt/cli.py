from __future__ import annotations

import argparse
from pathlib import Path

from .processor import process_directory
from .watcher import watch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="elect-mt", description="Automated image processing (OpenCV)")
    sub = parser.add_subparsers(dest="command", required=True)

    batch = sub.add_parser("batch", help="Process all images currently in the input directory")
    batch.add_argument("--input", dest="input_dir", type=Path, default=Path("input"))
    batch.add_argument("--output", dest="output_dir", type=Path, default=Path("output"))
    batch.add_argument("--recursive", action="store_true")
    batch.add_argument("--overwrite", action="store_true")

    watch_p = sub.add_parser("watch", help="Watch input directory and process new images")
    watch_p.add_argument("--input", dest="input_dir", type=Path, default=Path("input"))
    watch_p.add_argument("--output", dest="output_dir", type=Path, default=Path("output"))
    watch_p.add_argument("--recursive", action="store_true")
    watch_p.add_argument("--overwrite", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "batch":
        results = process_directory(
            args.input_dir,
            args.output_dir,
            recursive=args.recursive,
            overwrite=args.overwrite,
        )
        # Keep CLI output minimal and CI-friendly.
        print(f"Processed {len(results)} image(s)")
        return 0

    if args.command == "watch":
        watch(
            args.input_dir,
            args.output_dir,
            recursive=args.recursive,
            overwrite=args.overwrite,
        )
        return 0

    parser.error("Unknown command")
    return 2
