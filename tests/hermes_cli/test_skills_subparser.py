"""Test that skills subparser doesn't conflict (regression test for #898)."""

import argparse

from hermes_cli import main as cli_main


def test_no_duplicate_skills_subparser():
    """Ensure 'skills' subparser is only registered once.

    Python 3.11+ raises argparse.ArgumentError when the same subparser name is
    registered twice. Building a fresh CLI parser exercises the complete
    registration path without replacing hermes_cli.main in sys.modules.
    """
    try:
        cli_main._build_cli_parser()
    except argparse.ArgumentError as e:
        if "conflicting subparser" in str(e):
            raise AssertionError(
                f"Duplicate subparser detected: {e}. "
                "See issue #898 for details."
            ) from e
        raise
