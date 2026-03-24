"""Validation module for self-improvement changes.

Runs validation checks against a worktree to ensure changes won't break the agent.
Also enforces the selfdev-policy.yaml frozen file restrictions.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ValidationResult:
    passed: bool
    checks: list[CheckResult]

    @property
    def summary(self) -> str:
        lines = []
        for check in self.checks:
            icon = "PASS" if check.passed else "FAIL"
            lines.append(f"  [{icon}] {check.name}")
            if not check.passed and check.output:
                for line in check.output.strip().splitlines()[:5]:
                    lines.append(f"         {line}")
        return "\n".join(lines)


@dataclass
class CheckResult:
    name: str
    passed: bool
    output: str = ""


def load_policy(repo_root: Path) -> dict:
    policy_path = repo_root / "selfdev-policy.yaml"
    if not policy_path.exists():
        return {"frozen": [], "allowed": [], "unrestricted": [], "validation": []}
    with open(policy_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check_frozen_files(repo_root: Path, worktree_path: Path, policy: dict) -> CheckResult:
    """Ensure no frozen files were modified in the worktree."""
    frozen_patterns = policy.get("frozen", [])
    if not frozen_patterns:
        return CheckResult(name="frozen files", passed=True)

    result = subprocess.run(
        ["git", "diff", "--name-only", "main", "HEAD"],
        capture_output=True, text=True, cwd=worktree_path, check=False,
    )
    if result.returncode != 0:
        return CheckResult(name="frozen files", passed=False, output=result.stderr)

    changed_files = result.stdout.strip().splitlines()
    violations = []
    for changed in changed_files:
        for pattern in frozen_patterns:
            if _matches_pattern(changed, pattern):
                violations.append(changed)
                break

    if violations:
        return CheckResult(
            name="frozen files",
            passed=False,
            output=f"Frozen files were modified: {', '.join(violations)}",
        )
    return CheckResult(name="frozen files", passed=True)


def run_validation_commands(worktree_path: Path, policy: dict) -> list[CheckResult]:
    """Run the validation commands from the policy."""
    commands = policy.get("validation", [])
    results = []
    for command in commands:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            cwd=worktree_path, check=False, timeout=120,
        )
        passed = result.returncode == 0
        output = (result.stdout + result.stderr).strip()
        results.append(CheckResult(name=command, passed=passed, output=output))
        if not passed:
            break  # stop on first failure
    return results


def validate_worktree(repo_root: Path, worktree_path: Path) -> ValidationResult:
    """Run all validation checks against a worktree."""
    policy = load_policy(repo_root)

    checks: list[CheckResult] = []

    # Check frozen files
    frozen_check = check_frozen_files(repo_root, worktree_path, policy)
    checks.append(frozen_check)
    if not frozen_check.passed:
        return ValidationResult(passed=False, checks=checks)

    # Run validation commands
    command_checks = run_validation_commands(worktree_path, policy)
    checks.extend(command_checks)

    all_passed = all(c.passed for c in checks)
    return ValidationResult(passed=all_passed, checks=checks)


def _matches_pattern(filepath: str, pattern: str) -> bool:
    """Simple glob-style matching: ** matches any path segment, * matches within a segment."""
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        return filepath.startswith(prefix + "/") or filepath == prefix
    if pattern.endswith("/*"):
        prefix = pattern[:-2]
        return filepath.startswith(prefix + "/") and "/" not in filepath[len(prefix) + 1:]
    return filepath == pattern
