"""avoid_agent/__main__.py — thin CLI entry point.

This module is the main entry point for the avoid-agent CLI. It handles:
- Argument parsing
- Dispatching to the appropriate CLI mode (tui, headless, selfdev, prompt)

All actual logic is delegated to modules in avoid_agent.cli/
"""

from __future__ import annotations

import argparse
import sys

from avoid_agent import cli
from avoid_agent.infra import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for all CLI subcommands."""
    parser = argparse.ArgumentParser(prog="avoid-agent")
    subparsers = parser.add_subparsers(dest="command")

    # Prompt subcommand
    prompt_parser = subparsers.add_parser("prompt", help="Prompt development utilities")
    prompt_subparsers = prompt_parser.add_subparsers(dest="prompt_command")
    export_parser = prompt_subparsers.add_parser("export", help="Export system prompt to markdown")
    export_parser.add_argument(
        "--out",
        default="./system-prompt.md",
        help="Output markdown file path (default: ./system-prompt.md)",
    )

    # Headless subcommand
    headless_parser = subparsers.add_parser(
        "headless", help="Run agent in headless mode for programmatic use"
    )
    headless_parser.add_argument("--prompt", type=str, default=None, help="Single-turn prompt text")
    headless_parser.add_argument("--session", type=str, default=None, help="Session name for persistence")
    headless_parser.add_argument("--auto-approve", action="store_true", help="Auto-approve all bash commands")
    headless_parser.add_argument("--model", type=str, default=None, help="Provider/model (e.g. anthropic/claude-sonnet-4-6)")
    headless_parser.add_argument("--max-turns", type=int, default=20, help="Max turns in multi-turn stdin mode")
    headless_parser.add_argument("--context-strategy", type=str, default="compact+window", help="Context management strategy")
    headless_parser.add_argument("--no-session", action="store_true", help="Don't persist session (ephemeral run)")
    headless_parser.add_argument("--context-budget", type=int, default=None, help="Max input tokens for context")
    headless_parser.add_argument("--compaction-cooldown", type=int, default=3, help="Min turns between compactions")

    # Selfdev subcommand
    selfdev_parser = subparsers.add_parser("selfdev", help="Run the self-improvement loop")
    selfdev_parser.add_argument("--model", type=str, default=None, help="Provider/model for headless agent")
    selfdev_parser.add_argument("--max-turns", type=int, default=40, help="Max turns per headless run")
    selfdev_parser.add_argument("--single", action="store_true", help="Run only one cycle then exit")
    selfdev_parser.add_argument("--operator", action="store_true", help="Use operator agent mode")
    selfdev_parser.add_argument("--interactive", action="store_true", help="Use interactive mode with user feedback")
    selfdev_parser.add_argument("--legacy", action="store_true", help="Use legacy non-interactive selfdev loop output")

    # TUI mode (default - no subcommand needed)
    # The TUI is the default mode when no subcommand is specified

    return parser


def _export_prompt_command(output: str) -> None:
    """Export system prompt to markdown."""
    from pathlib import Path
    from avoid_agent.cli.shared import gather_initial_context
    from avoid_agent.prompts.system_prompt import SystemPromptOptions
    from avoid_agent.prompts.export_markdown import export_system_prompt_markdown

    cwd, git_status, top_level_structure = gather_initial_context()
    written = export_system_prompt_markdown(
        output,
        options=SystemPromptOptions(
            working_directory=cwd,
            git_status=git_status,
            top_level_file_structure=top_level_structure,
        ),
    )
    print(f"Exported system prompt markdown to: {written}")


def main() -> None:
    """Main entry point for the avoid-agent CLI."""
    setup_logging()
    parser = _build_parser()
    args = parser.parse_args()

    # Prompt export command
    if args.command == "prompt" and args.prompt_command == "export":
        _export_prompt_command(args.out)
        return

    # Headless mode
    if args.command == "headless":
        exit_code = cli.headless.run(args)
        sys.exit(exit_code)

    # Selfdev mode
    if args.command == "selfdev":
        exit_code = cli.selfdev.run(args)
        sys.exit(exit_code)

    # Default: TUI mode
    cli.tui.run(args)


if __name__ == "__main__":
    main()
