"""Minimal command-line entry point for the Battery-AAR prototype."""

from __future__ import annotations

import argparse

from battery_aar.protocols.policy_space import generate_protocol_space


def main() -> None:
    parser = argparse.ArgumentParser(description="Battery-AAR prototype utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    protocols = subparsers.add_parser("protocols", help="Print the protocol-space size")
    protocols.add_argument("--head", type=int, default=5, help="Rows to print")

    args = parser.parse_args()
    if args.command == "protocols":
        df = generate_protocol_space()
        print(f"protocol_count={len(df)}")
        print(df.head(args.head).to_string(index=False))
