---
name: Self Review
description: Review recent changes with git before finalizing work.
---

# Self Review

Use this skill before you report completion.

## Purpose
Perform a quick self-check of recent changes to catch mistakes and confirm the request is fully satisfied.

## Steps
1. Review recent commit context.
   - Run `git log --oneline -n 10`
2. Review local modifications.
   - Run `git diff`
   - If relevant, also run `git diff --staged`
3. Read every modified file directly.
   - Use `read_file` on each changed file; do not rely only on diff hunks.
4. Confirm alignment with the task.
   - Verify requested behavior is implemented.
   - Verify only allowed files were changed.
5. Run focused validation.
   - Run targeted tests/checks for the changed area.
6. Report clearly.
   - Summarize changes, validation results, and any remaining risks.
