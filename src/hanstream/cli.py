"""Local corpus inspection and transcript scoring command line interface."""

import argparse
import json
from dataclasses import asdict

from .manifest import corpus_summary, load_manifest
from .metrics import score


def main(argv=None) -> int:
    """Run a subcommand and print a machine-readable JSON result."""
    parser = argparse.ArgumentParser(prog='hanstream')
    commands = parser.add_subparsers(dest='command', required=True)
    scoring = commands.add_parser('score', help='score a pair of transcripts')
    scoring.add_argument('reference')
    scoring.add_argument('hypothesis')
    scoring.add_argument('--unit', choices=['word', 'char'], default='char')
    audit = commands.add_parser('audit', help='validate a JSONL corpus manifest')
    audit.add_argument('manifest')
    audit.add_argument('--check-audio', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'score':
            result = score(args.reference, args.hypothesis, args.unit)
            output = {**asdict(result), 'rate': result.rate}
        else:
            output = corpus_summary(load_manifest(args.manifest, args.check_audio))
    except (ValueError, OSError) as exc:
        parser.exit(2, f'error: {exc}\n')
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0
