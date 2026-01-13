"""Git utilities for extracting diffs and file context for code reviews."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DiffResult:
    """Result of a git diff operation."""

    diff: str
    files: list[str]
    title: str
    base_ref: Optional[str] = None
    head_ref: Optional[str] = None


def run_git_command(args: list[str], check: bool = True) -> tuple[str, str, int]:
    """Run a git command and return stdout, stderr, and return code.

    Args:
        args: Git command arguments (without 'git' prefix).
        check: If True, raise exception on non-zero exit.

    Returns:
        Tuple of (stdout, stderr, return_code).

    Raises:
        subprocess.CalledProcessError: If check=True and command fails.
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            check=check,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.CalledProcessError as e:
        if check:
            raise
        return e.stdout or "", e.stderr or "", e.returncode


def is_git_repo() -> bool:
    """Check if current directory is inside a git repository."""
    _, _, code = run_git_command(["rev-parse", "--git-dir"], check=False)
    return code == 0


def get_current_branch() -> Optional[str]:
    """Get the current branch name.

    Returns:
        Branch name, or None if in detached HEAD state.
    """
    stdout, _, code = run_git_command(
        ["rev-parse", "--abbrev-ref", "HEAD"], check=False
    )
    if code != 0:
        return None
    branch = stdout.strip()
    return None if branch == "HEAD" else branch


def get_default_branch() -> str:
    """Get the default branch name (main or master).

    Returns:
        Default branch name, defaults to 'main' if can't determine.
    """
    # Try to get from remote origin
    stdout, _, code = run_git_command(
        ["symbolic-ref", "refs/remotes/origin/HEAD"], check=False
    )
    if code == 0:
        # Output like: refs/remotes/origin/main
        return stdout.strip().split("/")[-1]

    # Fall back to checking if main or master exists
    for branch in ["main", "master"]:
        _, _, code = run_git_command(["rev-parse", "--verify", branch], check=False)
        if code == 0:
            return branch

    return "main"


def get_available_branches() -> list[str]:
    """Get list of available local and remote branches.

    Returns:
        List of branch names (local branches first, then remote).
    """
    branches = []

    # Local branches
    stdout, _, _ = run_git_command(["branch", "--format=%(refname:short)"], check=False)
    if stdout:
        branches.extend(stdout.strip().split("\n"))

    # Remote branches (excluding HEAD)
    stdout, _, _ = run_git_command(
        ["branch", "-r", "--format=%(refname:short)"], check=False
    )
    if stdout:
        for branch in stdout.strip().split("\n"):
            if branch and not branch.endswith("/HEAD"):
                branches.append(branch)

    return branches


def get_merge_base(base: str, head: str = "HEAD") -> Optional[str]:
    """Get the merge base commit between two refs.

    Args:
        base: Base reference (branch, tag, or commit).
        head: Head reference, defaults to HEAD.

    Returns:
        Merge base commit SHA, or None if not found.
    """
    stdout, _, code = run_git_command(["merge-base", base, head], check=False)
    if code != 0:
        return None
    return stdout.strip()


def get_branch_diff(base: str, head: str = "HEAD") -> DiffResult:
    """Get diff between a base branch and head.

    This is "PR style" - shows what would be merged.

    Args:
        base: Base branch/ref to compare against.
        head: Head ref, defaults to HEAD.

    Returns:
        DiffResult with diff content and changed files.

    Raises:
        ValueError: If base ref doesn't exist.
    """
    # Verify base exists
    _, _, code = run_git_command(["rev-parse", "--verify", base], check=False)
    if code != 0:
        raise ValueError(f"Base ref '{base}' not found")

    # Get merge base for proper PR-style diff
    merge_base = get_merge_base(base, head)
    if not merge_base:
        # Fall back to direct diff if no merge base
        merge_base = base

    # Get the diff
    stdout, stderr, code = run_git_command(
        ["diff", "--no-color", merge_base, head], check=False
    )
    if code != 0:
        raise ValueError(f"Failed to get diff: {stderr}")

    diff = stdout

    # Get list of changed files
    files_stdout, _, _ = run_git_command(
        ["diff", "--name-only", merge_base, head], check=False
    )
    files = [f for f in files_stdout.strip().split("\n") if f]

    # Get short refs for title
    head_name = head
    if head == "HEAD":
        head_name = get_current_branch() or "HEAD"

    return DiffResult(
        diff=diff,
        files=files,
        title=f"Changes from {base} to {head_name}",
        base_ref=base,
        head_ref=head,
    )


def get_uncommitted_diff(staged_only: bool = False) -> DiffResult:
    """Get diff of uncommitted changes.

    Args:
        staged_only: If True, only include staged changes.

    Returns:
        DiffResult with diff content and changed files.
    """
    if staged_only:
        # Only staged changes
        stdout, _, _ = run_git_command(["diff", "--cached", "--no-color"], check=False)
        files_stdout, _, _ = run_git_command(
            ["diff", "--cached", "--name-only"], check=False
        )
        title = "Staged changes"
    else:
        # All uncommitted changes (staged + unstaged)
        # First get staged
        staged_diff, _, _ = run_git_command(
            ["diff", "--cached", "--no-color"], check=False
        )
        staged_files, _, _ = run_git_command(
            ["diff", "--cached", "--name-only"], check=False
        )

        # Then get unstaged
        unstaged_diff, _, _ = run_git_command(["diff", "--no-color"], check=False)
        unstaged_files, _, _ = run_git_command(["diff", "--name-only"], check=False)

        # Combine
        stdout = ""
        if staged_diff:
            stdout += "# Staged changes\n" + staged_diff
        if unstaged_diff:
            if stdout:
                stdout += "\n\n"
            stdout += "# Unstaged changes\n" + unstaged_diff

        files_stdout = staged_files + "\n" + unstaged_files
        title = "Uncommitted changes"

    files = list(set(f for f in files_stdout.strip().split("\n") if f))

    return DiffResult(
        diff=stdout,
        files=files,
        title=title,
    )


def get_commit_diff(commit: str) -> DiffResult:
    """Get diff for a specific commit.

    Args:
        commit: Commit SHA or ref.

    Returns:
        DiffResult with diff content and changed files.

    Raises:
        ValueError: If commit doesn't exist.
    """
    # Verify commit exists
    _, stderr, code = run_git_command(["rev-parse", "--verify", commit], check=False)
    if code != 0:
        raise ValueError(f"Commit '{commit}' not found: {stderr}")

    # Get the diff (commit vs its parent)
    stdout, stderr, code = run_git_command(
        ["show", "--no-color", "--format=", commit], check=False
    )
    if code != 0:
        raise ValueError(f"Failed to get diff for commit: {stderr}")

    diff = stdout

    # Get list of changed files
    files_stdout, _, _ = run_git_command(
        ["diff-tree", "--no-commit-id", "--name-only", "-r", commit], check=False
    )
    files = [f for f in files_stdout.strip().split("\n") if f]

    # Get commit message for title
    msg_stdout, _, _ = run_git_command(
        ["log", "-1", "--format=%s", commit], check=False
    )
    commit_msg = msg_stdout.strip()[:50]

    # Get short SHA
    short_sha, _, _ = run_git_command(["rev-parse", "--short", commit], check=False)
    short_sha = short_sha.strip()

    return DiffResult(
        diff=diff,
        files=files,
        title=f"Commit {short_sha}: {commit_msg}",
        head_ref=commit,
    )


def get_recent_commits(count: int = 10) -> list[dict]:
    """Get list of recent commits for selection.

    Args:
        count: Number of commits to return.

    Returns:
        List of dicts with sha, short_sha, message, author, date.
    """
    stdout, _, code = run_git_command(
        [
            "log",
            f"-{count}",
            "--format=%H|%h|%s|%an|%ar",
        ],
        check=False,
    )
    if code != 0:
        return []

    commits = []
    for line in stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|", 4)
        if len(parts) >= 5:
            commits.append(
                {
                    "sha": parts[0],
                    "short_sha": parts[1],
                    "message": parts[2][:60],
                    "author": parts[3],
                    "date": parts[4],
                }
            )
    return commits


def get_file_content(file_path: str, ref: Optional[str] = None) -> Optional[str]:
    """Get content of a file, optionally at a specific ref.

    Args:
        file_path: Path to the file.
        ref: Git ref (branch, commit) to get file from. None for working tree.

    Returns:
        File content, or None if file doesn't exist.
    """
    if ref:
        stdout, _, code = run_git_command(["show", f"{ref}:{file_path}"], check=False)
        if code != 0:
            return None
        return stdout
    else:
        path = Path(file_path)
        if not path.exists():
            return None
        try:
            return path.read_text()
        except Exception:
            return None


def get_file_with_line_numbers(file_path: str, ref: Optional[str] = None) -> str:
    """Get file content with line numbers for context.

    Args:
        file_path: Path to the file.
        ref: Git ref to get file from. None for working tree.

    Returns:
        File content with line numbers, or error message.
    """
    content = get_file_content(file_path, ref)
    if content is None:
        return f"# Error: Could not read {file_path}\n"

    lines = content.split("\n")
    numbered_lines = []
    width = len(str(len(lines)))
    for i, line in enumerate(lines, 1):
        numbered_lines.append(f"{i:>{width}} | {line}")

    return f"# {file_path}\n" + "\n".join(numbered_lines)


def get_diff_stats(diff: str) -> dict:
    """Parse diff to get statistics.

    Args:
        diff: Diff content.

    Returns:
        Dict with insertions, deletions, files_changed.
    """
    insertions = 0
    deletions = 0
    files = set()

    for line in diff.split("\n"):
        if line.startswith("+++ b/"):
            files.add(line[6:])
        elif line.startswith("+") and not line.startswith("+++"):
            insertions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1

    return {
        "insertions": insertions,
        "deletions": deletions,
        "files_changed": len(files),
    }


def format_branch_choices(current_branch: Optional[str] = None) -> list[dict]:
    """Format branch choices for PR-style review selection.

    Args:
        current_branch: Current branch name for comparison display.

    Returns:
        List of dicts with value and display for each comparison option.
    """
    if not current_branch:
        current_branch = get_current_branch()

    default = get_default_branch()
    branches = get_available_branches()

    choices = []

    # Default comparison first
    if default in branches:
        choices.append(
            {
                "value": default,
                "display": f"{current_branch} -> {default}",
                "is_default": True,
            }
        )

    # Other local branches
    for branch in branches:
        if branch == default or branch == current_branch:
            continue
        if "/" in branch:  # Skip remote branches in main list
            continue
        choices.append(
            {
                "value": branch,
                "display": f"{current_branch} -> {branch}",
                "is_default": False,
            }
        )

    return choices


def build_review_document(
    diff_result: DiffResult,
    file_context: Optional[dict[str, str]] = None,
    custom_instructions: Optional[str] = None,
) -> str:
    """Build a document for code review from diff and context.

    Args:
        diff_result: Result from get_*_diff() functions.
        file_context: Optional dict of file_path -> full file content.
        custom_instructions: Optional custom review instructions.

    Returns:
        Formatted document ready for code review.
    """
    stats = get_diff_stats(diff_result.diff)

    doc = f"""# Code Review: {diff_result.title}

## Overview
- Files changed: {stats['files_changed']}
- Lines added: {stats['insertions']}
- Lines removed: {stats['deletions']}

## Changed Files
{chr(10).join(f'- {f}' for f in diff_result.files)}

"""

    if custom_instructions:
        doc += f"""## Review Instructions
{custom_instructions}

"""

    doc += f"""## Diff
```diff
{diff_result.diff}
```

"""

    if file_context:
        doc += "## Full File Context\n\n"
        for path, content in file_context.items():
            doc += f"### {path}\n```\n{content}\n```\n\n"

    return doc
