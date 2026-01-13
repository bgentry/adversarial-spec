"""Tests for git_utils module."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from git_utils import (
    DiffResult,
    run_git_command,
    is_git_repo,
    get_current_branch,
    get_default_branch,
    get_available_branches,
    get_merge_base,
    get_branch_diff,
    get_uncommitted_diff,
    get_commit_diff,
    get_recent_commits,
    get_file_content,
    get_file_with_line_numbers,
    get_diff_stats,
    format_branch_choices,
    build_review_document,
)


class TestDiffResult:
    def test_create_diff_result(self):
        result = DiffResult(
            diff="diff --git a/file.py",
            files=["file.py"],
            title="Test diff",
        )
        assert result.diff == "diff --git a/file.py"
        assert result.files == ["file.py"]
        assert result.title == "Test diff"
        assert result.base_ref is None
        assert result.head_ref is None

    def test_with_refs(self):
        result = DiffResult(
            diff="diff content",
            files=["a.py", "b.py"],
            title="Changes from main to feature",
            base_ref="main",
            head_ref="feature",
        )
        assert result.base_ref == "main"
        assert result.head_ref == "feature"


class TestRunGitCommand:
    @patch("subprocess.run")
    def test_successful_command(self, mock_run):
        mock_run.return_value = Mock(
            stdout="output",
            stderr="",
            returncode=0,
        )
        stdout, stderr, code = run_git_command(["status"])
        assert stdout == "output"
        assert stderr == ""
        assert code == 0
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_failed_command_no_check(self, mock_run):
        mock_run.return_value = Mock(
            stdout="",
            stderr="error",
            returncode=1,
        )
        stdout, stderr, code = run_git_command(["bad-command"], check=False)
        assert stderr == "error"
        assert code == 1


class TestIsGitRepo:
    @patch("git_utils.run_git_command")
    def test_is_git_repo_true(self, mock_run):
        mock_run.return_value = (".git", "", 0)
        assert is_git_repo() is True

    @patch("git_utils.run_git_command")
    def test_is_git_repo_false(self, mock_run):
        mock_run.return_value = ("", "not a git repo", 128)
        assert is_git_repo() is False


class TestGetCurrentBranch:
    @patch("git_utils.run_git_command")
    def test_on_branch(self, mock_run):
        mock_run.return_value = ("main\n", "", 0)
        assert get_current_branch() == "main"

    @patch("git_utils.run_git_command")
    def test_detached_head(self, mock_run):
        mock_run.return_value = ("HEAD\n", "", 0)
        assert get_current_branch() is None

    @patch("git_utils.run_git_command")
    def test_error(self, mock_run):
        mock_run.return_value = ("", "error", 1)
        assert get_current_branch() is None


class TestGetDefaultBranch:
    @patch("git_utils.run_git_command")
    def test_from_remote(self, mock_run):
        mock_run.return_value = ("refs/remotes/origin/main\n", "", 0)
        assert get_default_branch() == "main"

    @patch("git_utils.run_git_command")
    def test_fallback_to_main(self, mock_run):
        def side_effect(args, check=True):
            if "symbolic-ref" in args:
                return ("", "error", 1)
            if args == ["rev-parse", "--verify", "main"]:
                return ("sha", "", 0)
            return ("", "error", 1)

        mock_run.side_effect = side_effect
        assert get_default_branch() == "main"


class TestGetAvailableBranches:
    @patch("git_utils.run_git_command")
    def test_local_and_remote(self, mock_run):
        def side_effect(args, check=True):
            if "-r" in args:
                return ("origin/main\norigin/HEAD\n", "", 0)
            return ("main\nfeature\n", "", 0)

        mock_run.side_effect = side_effect
        branches = get_available_branches()
        assert "main" in branches
        assert "feature" in branches
        assert "origin/main" in branches
        assert "origin/HEAD" not in branches  # Should be filtered


class TestGetMergeBase:
    @patch("git_utils.run_git_command")
    def test_found(self, mock_run):
        mock_run.return_value = ("abc123\n", "", 0)
        assert get_merge_base("main", "HEAD") == "abc123"

    @patch("git_utils.run_git_command")
    def test_not_found(self, mock_run):
        mock_run.return_value = ("", "error", 1)
        assert get_merge_base("main", "HEAD") is None


class TestGetBranchDiff:
    @patch("git_utils.get_current_branch")
    @patch("git_utils.get_merge_base")
    @patch("git_utils.run_git_command")
    def test_successful_diff(self, mock_run, mock_merge_base, mock_branch):
        mock_branch.return_value = "feature"
        mock_merge_base.return_value = "abc123"

        def side_effect(args, check=True):
            if "rev-parse" in args and "--verify" in args:
                return ("sha", "", 0)
            if "diff" in args and "--no-color" in args:
                return ("diff content", "", 0)
            if "diff" in args and "--name-only" in args:
                return ("file1.py\nfile2.py\n", "", 0)
            return ("", "", 0)

        mock_run.side_effect = side_effect

        result = get_branch_diff("main")
        assert result.diff == "diff content"
        assert result.files == ["file1.py", "file2.py"]
        assert "main" in result.title

    @patch("git_utils.run_git_command")
    def test_invalid_base(self, mock_run):
        mock_run.return_value = ("", "not found", 128)
        try:
            get_branch_diff("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not found" in str(e)


class TestGetUncommittedDiff:
    @patch("git_utils.run_git_command")
    def test_staged_and_unstaged(self, mock_run):
        def side_effect(args, check=True):
            if "--cached" in args and "--no-color" in args:
                return ("staged diff", "", 0)
            if "--cached" in args and "--name-only" in args:
                return ("staged.py\n", "", 0)
            if "--no-color" in args:
                return ("unstaged diff", "", 0)
            if "--name-only" in args:
                return ("unstaged.py\n", "", 0)
            return ("", "", 0)

        mock_run.side_effect = side_effect

        result = get_uncommitted_diff()
        assert "staged diff" in result.diff
        assert "unstaged diff" in result.diff
        assert "staged.py" in result.files
        assert "unstaged.py" in result.files

    @patch("git_utils.run_git_command")
    def test_staged_only(self, mock_run):
        mock_run.return_value = ("staged content", "", 0)

        result = get_uncommitted_diff(staged_only=True)
        assert "Staged changes" in result.title


class TestGetCommitDiff:
    @patch("git_utils.run_git_command")
    def test_successful_diff(self, mock_run):
        def side_effect(args, check=True):
            if "rev-parse" in args and "--verify" in args:
                return ("fullsha", "", 0)
            if "rev-parse" in args and "--short" in args:
                return ("abc123\n", "", 0)
            if "show" in args:
                return ("commit diff", "", 0)
            if "diff-tree" in args:
                return ("file.py\n", "", 0)
            if "log" in args:
                return ("Commit message\n", "", 0)
            return ("", "", 0)

        mock_run.side_effect = side_effect

        result = get_commit_diff("abc123")
        assert result.diff == "commit diff"
        assert "file.py" in result.files
        assert "abc123" in result.title

    @patch("git_utils.run_git_command")
    def test_invalid_commit(self, mock_run):
        mock_run.return_value = ("", "not found", 128)
        try:
            get_commit_diff("invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not found" in str(e)


class TestGetRecentCommits:
    @patch("git_utils.run_git_command")
    def test_parse_commits(self, mock_run):
        mock_run.return_value = (
            "abc123|abc|Fix bug|Author|2 hours ago\n"
            "def456|def|Add feature|Author|1 day ago\n",
            "",
            0,
        )
        commits = get_recent_commits(10)
        assert len(commits) == 2
        assert commits[0]["sha"] == "abc123"
        assert commits[0]["message"] == "Fix bug"
        assert commits[1]["short_sha"] == "def"


class TestGetFileContent:
    @patch("git_utils.run_git_command")
    def test_from_ref(self, mock_run):
        mock_run.return_value = ("file content", "", 0)
        content = get_file_content("file.py", "HEAD")
        assert content == "file content"

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.read_text")
    def test_from_working_tree(self, mock_read, mock_exists):
        mock_exists.return_value = True
        mock_read.return_value = "local content"
        content = get_file_content("file.py")
        assert content == "local content"

    @patch("pathlib.Path.exists")
    def test_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        content = get_file_content("missing.py")
        assert content is None


class TestGetFileWithLineNumbers:
    @patch("git_utils.get_file_content")
    def test_with_numbers(self, mock_content):
        mock_content.return_value = "line1\nline2\nline3"
        result = get_file_with_line_numbers("test.py")
        assert "# test.py" in result
        assert "1 | line1" in result
        assert "2 | line2" in result
        assert "3 | line3" in result

    @patch("git_utils.get_file_content")
    def test_file_error(self, mock_content):
        mock_content.return_value = None
        result = get_file_with_line_numbers("missing.py")
        assert "Error" in result


class TestGetDiffStats:
    def test_parse_stats(self):
        diff = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
+new line
 unchanged
-removed
+modified"""
        stats = get_diff_stats(diff)
        assert stats["insertions"] == 2
        assert stats["deletions"] == 1
        assert stats["files_changed"] == 1


class TestFormatBranchChoices:
    @patch("git_utils.get_current_branch")
    @patch("git_utils.get_default_branch")
    @patch("git_utils.get_available_branches")
    def test_format_choices(self, mock_branches, mock_default, mock_current):
        mock_current.return_value = "feature"
        mock_default.return_value = "main"
        mock_branches.return_value = ["main", "feature", "develop"]

        choices = format_branch_choices()
        assert len(choices) >= 1
        assert choices[0]["value"] == "main"
        assert choices[0]["is_default"] is True
        assert "feature -> main" in choices[0]["display"]


class TestBuildReviewDocument:
    def test_basic_document(self):
        diff_result = DiffResult(
            diff="diff content",
            files=["file.py"],
            title="Test Review",
        )
        doc = build_review_document(diff_result)
        assert "# Code Review: Test Review" in doc
        assert "file.py" in doc
        assert "diff content" in doc

    def test_with_context(self):
        diff_result = DiffResult(
            diff="diff",
            files=["a.py"],
            title="Review",
        )
        file_context = {"a.py": "def foo():\n    pass"}
        doc = build_review_document(diff_result, file_context)
        assert "Full File Context" in doc
        assert "def foo()" in doc

    def test_with_instructions(self):
        diff_result = DiffResult(
            diff="diff",
            files=["a.py"],
            title="Review",
        )
        doc = build_review_document(
            diff_result,
            custom_instructions="Focus on security",
        )
        assert "Review Instructions" in doc
        assert "Focus on security" in doc
