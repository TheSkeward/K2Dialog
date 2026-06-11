from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .dumper import DumpOptions, dump_game


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="k2dialog_dump")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump_parser = subparsers.add_parser("dump", help="Dump resolved KOTOR II dialogue to Markdown")
    dump_parser.add_argument("--game-dir", required=True, type=Path, help="KOTOR II install directory")
    dump_parser.add_argument("--out", required=True, type=Path, help="Output directory")
    dump_parser.add_argument("--single-file", action="store_true", help="Write out/all_dialogue.md")
    dump_parser.add_argument("--by-module", action="store_true", help="Write one Markdown file per module")
    dump_parser.add_argument("--by-dlg", action="store_true", help="Write one Markdown file per DLG")
    dump_parser.add_argument(
        "--show-unresolved-checks",
        action="store_true",
        help="Annotate skill-tagged options whose DC/check script is not yet decoded.",
    )
    dump_parser.add_argument("--quiet", action="store_true", help="Hide non-fatal parse warnings")

    args = parser.parse_args(argv)
    if args.command == "dump":
        logging.basicConfig(
            level=logging.ERROR if args.quiet else logging.WARNING,
            format="%(levelname)s: %(message)s",
        )
        any_output_flag = args.single_file or args.by_module or args.by_dlg
        options = DumpOptions(
            game_dir=args.game_dir,
            out_dir=args.out,
            single_file=args.single_file or not any_output_flag,
            by_module=args.by_module or not any_output_flag,
            by_dlg=args.by_dlg or not any_output_flag,
            show_unresolved_checks=args.show_unresolved_checks,
        )
        try:
            dumped = dump_game(options)
        except (FileNotFoundError, OSError, ValueError) as exc:
            parser.exit(1, f"error: {exc}\n")
        print(f"Dumped {len(dumped)} dialogue files to {options.out_dir.resolve()}")
        return 0
    return 2
