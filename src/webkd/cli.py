"""Command-line entry points for the WebKD pipeline."""

import argparse
import importlib
import sys

COMMANDS = {
    "edit": ("edit", "Generate WebKD edits or a named editing baseline"),
    "edit-offline": ("edit_offline", "Construct offline teacher or self-corrections"),
    "filter": ("filter", "Apply state-consistency filtering"),
    "convert": ("convert", "Convert trajectory steps to ShareGPT"),
    "prepare-scoring": ("prepare_scoring", "Prepare rationale ablation requests"),
    "score": ("score", "Score action likelihoods under a frozen student"),
    "build-weights": ("build_weights", "Build action-informativeness weights"),
    "build-anchor": ("build_anchor", "Build the rationale anchor mask"),
    "train": ("train", "Train WebKD or supervised fine-tuning"),
}


def main():
    parser = argparse.ArgumentParser(prog="webkd", description=__doc__)
    parser.add_argument("command", choices=COMMANDS, help="Pipeline stage")
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        for command, (_, description) in COMMANDS.items():
            print(f"  {command:18} {description}")
        return
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [f"webkd {args.command}", *sys.argv[2:]]
    importlib.import_module(f"webkd.{COMMANDS[args.command][0]}").main()
