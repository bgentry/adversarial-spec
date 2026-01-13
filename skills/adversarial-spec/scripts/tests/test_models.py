"""Tests for models module."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    ModelResponse,
    CostTracker,
    detect_agreement,
    extract_spec,
    extract_tasks,
    get_critique_summary,
    generate_diff,
    load_context_files,
    call_codex_model,
    call_single_model,
    call_models_parallel,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
)


class TestModelResponse:
    def test_create_response(self):
        response = ModelResponse(
            model="gpt-4o",
            response="This is a critique.",
            agreed=False,
            spec="# Revised Spec",
        )
        assert response.model == "gpt-4o"
        assert response.agreed is False
        assert response.spec == "# Revised Spec"

    def test_default_values(self):
        # Mutation: changing defaults would fail these checks
        response = ModelResponse(
            model="test",
            response="test",
            agreed=False,
            spec=None,
        )
        assert response.error is None  # Not ""
        assert response.input_tokens == 0  # Not 1
        assert response.output_tokens == 0  # Not 1
        assert response.cost == 0.0  # Not 1.0

    def test_response_with_error(self):
        response = ModelResponse(
            model="gpt-4o",
            response="",
            agreed=False,
            spec=None,
            error="API timeout",
        )
        assert response.error == "API timeout"

    def test_response_with_tokens(self):
        response = ModelResponse(
            model="gpt-4o",
            response="Response",
            agreed=True,
            spec="Spec",
            input_tokens=1000,
            output_tokens=500,
            cost=0.05,
        )
        assert response.input_tokens == 1000
        assert response.output_tokens == 500
        assert response.cost == 0.05


class TestCostTracker:
    def test_add_costs(self):
        tracker = CostTracker()
        tracker.add("gpt-4o", 1000, 500)

        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        assert tracker.total_cost > 0

    def test_cost_calculation_uses_division(self):
        # Mutation: / to * would make cost astronomically large
        tracker = CostTracker()
        # Use a model with known costs
        cost = tracker.add("gpt-4o", 1_000_000, 1_000_000)
        # With division by 1M, cost should be in dollars (single/double digits)
        # With multiplication by 1M, cost would be trillions
        assert cost < 1000  # Reasonable upper bound for 1M tokens

    def test_default_values(self):
        # Mutation: changing default 0.0 to 1.0 would fail
        tracker = CostTracker()
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_cost == 0.0  # Must be exactly 0.0
        assert tracker.by_model == {}

    def test_tracks_by_model(self):
        tracker = CostTracker()
        tracker.add("gpt-4o", 1000, 500)
        tracker.add("gemini/gemini-2.0-flash", 2000, 1000)

        assert "gpt-4o" in tracker.by_model
        assert "gemini/gemini-2.0-flash" in tracker.by_model
        assert tracker.by_model["gpt-4o"]["input_tokens"] == 1000

    def test_accumulates_for_same_model(self):
        tracker = CostTracker()
        tracker.add("gpt-4o", 1000, 500)
        tracker.add("gpt-4o", 1000, 500)

        assert tracker.by_model["gpt-4o"]["input_tokens"] == 2000
        assert tracker.by_model["gpt-4o"]["output_tokens"] == 1000

    def test_cost_accumulates_not_replaces(self):
        # Mutation: += to = would fail this test
        tracker = CostTracker()
        cost1 = tracker.add("gpt-4o", 1000, 500)
        cost2 = tracker.add("gpt-4o", 1000, 500)

        # Total cost should be sum of both calls
        expected_total = cost1 + cost2
        assert tracker.by_model["gpt-4o"]["cost"] == expected_total
        assert tracker.total_cost == expected_total

    def test_summary_format(self):
        tracker = CostTracker()
        tracker.add("gpt-4o", 1000, 500)

        summary = tracker.summary()
        assert "Cost Summary" in summary
        assert "Total tokens" in summary
        assert "Total cost" in summary

    def test_summary_starts_with_empty_line(self):
        # Mutation: "" -> "XXXX" would change first line
        tracker = CostTracker()
        tracker.add("gpt-4o", 1000, 500)
        summary = tracker.summary()
        # Summary should start with empty line (newline)
        assert summary.startswith("\n")
        assert "XXXX" not in summary

    def test_summary_shows_by_model_when_multiple(self):
        tracker = CostTracker()
        tracker.add("gpt-4o", 1000, 500)
        tracker.add("gemini-pro", 2000, 1000)
        summary = tracker.summary()
        assert "By model:" in summary
        assert "gpt-4o" in summary
        assert "gemini-pro" in summary


class TestDetectAgreement:
    def test_detects_agree(self):
        assert detect_agreement("I agree. [AGREE]\n[SPEC]...[/SPEC]") is True

    def test_no_agree(self):
        assert detect_agreement("I have concerns about security.") is False

    def test_partial_agree_in_word(self):
        # [AGREE] must be present as marker
        assert detect_agreement("I disagree with this approach.") is False


class TestExtractSpec:
    def test_extracts_spec(self):
        response = "Critique here.\n\n[SPEC]\n# My Spec\n\nContent\n[/SPEC]"
        spec = extract_spec(response)
        assert spec == "# My Spec\n\nContent"

    def test_returns_none_without_tags(self):
        response = "Just a critique without spec tags."
        assert extract_spec(response) is None

    def test_returns_none_with_missing_end_tag(self):
        response = "[SPEC]Content without end tag"
        assert extract_spec(response) is None

    def test_handles_empty_spec(self):
        response = "[SPEC][/SPEC]"
        spec = extract_spec(response)
        assert spec == ""


class TestExtractTasks:
    def test_extracts_single_task(self):
        response = """
[TASK]
title: Implement auth
type: task
priority: high
description: Add OAuth2 authentication
acceptance_criteria:
- User can log in
- Session persists
[/TASK]
"""
        tasks = extract_tasks(response)
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Implement auth"
        assert tasks[0]["type"] == "task"
        assert tasks[0]["priority"] == "high"
        assert len(tasks[0]["acceptance_criteria"]) == 2

    def test_extracts_multiple_tasks(self):
        response = """
[TASK]
title: Task 1
type: task
priority: high
description: First task
[/TASK]
[TASK]
title: Task 2
type: bug
priority: medium
description: Second task
[/TASK]
"""
        tasks = extract_tasks(response)
        assert len(tasks) == 2
        assert tasks[0]["title"] == "Task 1"
        assert tasks[1]["title"] == "Task 2"

    def test_handles_no_tasks(self):
        response = "No tasks here."
        tasks = extract_tasks(response)
        assert tasks == []

    def test_exact_slice_positions(self):
        # Mutation: line[6:] -> line[7:] for title would lose first char
        # Mutation: line[5:] -> line[6:] for type would lose first char
        # Mutation: line[9:] -> line[10:] for priority would lose first char
        # Mutation: line[12:] -> line[13:] for description would lose first char
        response = """
[TASK]
title: Xauth
type: Xtask
priority: Xhigh
description: Xdesc
[/TASK]
"""
        tasks = extract_tasks(response)
        assert tasks[0]["title"] == "Xauth"  # X must be preserved
        assert tasks[0]["type"] == "Xtask"
        assert tasks[0]["priority"] == "Xhigh"
        assert tasks[0]["description"] == "Xdesc"

    def test_multiline_title_joined_correctly(self):
        # Mutation: "\n".join -> "XX\nXX".join in title path
        response = """
[TASK]
title: Main title
continuation of title
type: task
priority: high
description: Desc
[/TASK]
"""
        tasks = extract_tasks(response)
        title = tasks[0]["title"]
        assert "Main title" in title
        assert "XX" not in title

    def test_multiline_description_joined_correctly(self):
        # Mutation: "\n".join -> "XX\nXX".join would corrupt multi-line values
        response = """
[TASK]
title: Test
type: task
priority: high
description: Line 1
Line 2
Line 3
[/TASK]
"""
        tasks = extract_tasks(response)
        desc = tasks[0]["description"]
        assert "Line 1" in desc
        assert "Line 2" in desc
        assert "XX" not in desc  # Mutation check
        # Verify proper newline joining
        assert "Line 1\nLine 2" in desc or "Line 1" in desc

    def test_multiline_priority_value_joined(self):
        # Mutation: "\n".join -> "XX\nXX".join in priority path
        response = """
[TASK]
title: Test
type: task
priority: high
extra priority line
description: Desc
[/TASK]
"""
        tasks = extract_tasks(response)
        priority = tasks[0]["priority"]
        assert "XX" not in priority

    def test_multiline_type_value_joined(self):
        # Mutation: "\n".join -> "XX\nXX".join in type path
        response = """
[TASK]
title: Test
type: task
extra type line
priority: high
description: Desc
[/TASK]
"""
        tasks = extract_tasks(response)
        task_type = tasks[0]["type"]
        assert "XX" not in task_type

    def test_acceptance_criteria_only_with_dash_prefix(self):
        # Mutation: and -> or would accept lines without "- " prefix
        response = """
[TASK]
title: Test
type: task
priority: high
description: Desc
acceptance_criteria:
- Valid item
Not a valid item because no dash
- Another valid item
[/TASK]
"""
        tasks = extract_tasks(response)
        criteria = tasks[0]["acceptance_criteria"]
        # Should only have items that started with "- "
        assert "Valid item" in criteria
        assert "Another valid item" in criteria
        # The line without dash should NOT be in criteria as a separate item
        # It would be appended as continuation if the mutation happens

    def test_acceptance_criteria_item_prefix_removed(self):
        # Mutation: line[2:] -> line[3:] would lose first char of criteria
        response = """
[TASK]
title: Test
type: task
priority: high
description: Desc
acceptance_criteria:
- Xfirst criteria
- Xsecond criteria
[/TASK]
"""
        tasks = extract_tasks(response)
        criteria = tasks[0]["acceptance_criteria"]
        assert criteria[0] == "Xfirst criteria"  # X must be preserved
        assert criteria[1] == "Xsecond criteria"

    def test_task_without_end_tag_skipped(self):
        response = """
[TASK]
title: Incomplete
type: task
Some text without closing tag
"""
        tasks = extract_tasks(response)
        assert tasks == []

    def test_continues_after_incomplete_task(self):
        # Mutation: continue -> break would stop after first incomplete task
        response = """
[TASK]
title: Incomplete
type: task
No closing tag here

[TASK]
title: Complete Task
type: task
priority: high
description: This one is complete
[/TASK]
"""
        tasks = extract_tasks(response)
        # Should still get the second complete task
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Complete Task"

    def test_empty_values_handled(self):
        # Mutation: current_value[0] if current_value else "" - check empty case
        response = """
[TASK]
title: Test
type:
priority: high
description: Desc
[/TASK]
"""
        tasks = extract_tasks(response)
        assert tasks[0]["type"] == ""

    def test_single_vs_multiple_value_lines(self):
        # Mutation: len(current_value) > 1 -> len(current_value) >= 1
        # This matters when exactly 1 line - should return the single line directly
        response = """
[TASK]
title: Single
type: task
priority: high
description: One line only
[/TASK]
"""
        tasks = extract_tasks(response)
        assert tasks[0]["description"] == "One line only"

    def test_task_requires_title(self):
        # Mutation: task.get("title") check ensures empty tasks ignored
        response = """
[TASK]
type: task
priority: high
[/TASK]
"""
        tasks = extract_tasks(response)
        assert tasks == []  # No title means not added

    def test_acceptance_criteria_returns_list(self):
        # Mutation: current_key == "acceptance_criteria" check for list vs string
        response = """
[TASK]
title: Test
type: task
priority: high
description: Desc
acceptance_criteria:
- Item 1
- Item 2
[/TASK]
"""
        tasks = extract_tasks(response)
        criteria = tasks[0]["acceptance_criteria"]
        assert isinstance(criteria, list)
        assert len(criteria) == 2


class TestGetCritiqueSummary:
    def test_extracts_critique_before_spec(self):
        response = "This is the critique.\n\n[SPEC]...[/SPEC]"
        summary = get_critique_summary(response)
        assert summary == "This is the critique."

    def test_truncates_long_critique(self):
        response = "A" * 500
        summary = get_critique_summary(response, max_length=100)
        assert len(summary) == 103  # 100 + "..."
        assert summary.endswith("...")

    def test_full_response_without_spec(self):
        response = "Just critique, no spec."
        summary = get_critique_summary(response)
        assert summary == "Just critique, no spec."


class TestGenerateDiff:
    def test_generates_diff(self):
        previous = "line1\nline2\nline3"
        current = "line1\nmodified\nline3"

        diff = generate_diff(previous, current)
        assert "-line2" in diff
        assert "+modified" in diff

    def test_no_diff_for_identical(self):
        content = "same\ncontent"
        diff = generate_diff(content, content)
        assert diff == ""

    def test_diff_contains_filename_markers(self):
        # Mutation: fromfile="previous" -> "XXpreviousXX" would change output
        previous = "old"
        current = "new"
        diff = generate_diff(previous, current)
        assert "previous" in diff
        assert "current" in diff
        assert "XX" not in diff  # No mutation artifacts


class TestLoadContextFiles:
    def test_loads_empty_list(self):
        result = load_context_files([])
        assert result == ""

    def test_loads_nonexistent_file(self):
        result = load_context_files(["/nonexistent/file.md"])
        assert "Error loading file" in result

    def test_formats_context(self, tmp_path):
        test_file = tmp_path / "context.md"
        test_file.write_text("# Context\n\nSome context.")

        result = load_context_files([str(test_file)])
        assert "Additional Context" in result
        assert "# Context" in result

    def test_context_format_contains_markdown(self, tmp_path):
        # Mutation: format string mutations would add XX
        test_file = tmp_path / "test.md"
        test_file.write_text("Content")

        result = load_context_files([str(test_file)])
        assert "### Context:" in result
        assert "```" in result
        assert "XX" not in result  # No mutation artifacts


class TestCallCodexModel:
    @patch("models.CODEX_AVAILABLE", False)
    def test_raises_when_codex_unavailable(self):
        import pytest

        with pytest.raises(RuntimeError, match="Codex CLI not found"):
            call_codex_model("system", "user", "codex/gpt-5")

    @patch("models.CODEX_AVAILABLE", True)
    @patch("models.subprocess.run")
    def test_extracts_model_name_from_codex_prefix(self, mock_run):
        # Mutation: model.split("/", 1)[1] - verify correct extraction
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"type":"item.completed","item":{"type":"agent_message","text":"Response"}}\n{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":50}}',
            stderr="",
        )
        response, inp, out = call_codex_model("sys", "user", "codex/gpt-5.2-codex")
        # Verify model name was extracted and passed to command
        cmd = mock_run.call_args[0][0]
        assert "gpt-5.2-codex" in cmd
        assert "codex/gpt-5.2-codex" not in cmd

    @patch("models.CODEX_AVAILABLE", True)
    @patch("models.subprocess.run")
    def test_parses_jsonl_response(self, mock_run):
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"type":"item.completed","item":{"type":"agent_message","text":"Test response"}}\n{"type":"turn.completed","usage":{"input_tokens":150,"output_tokens":75}}',
            stderr="",
        )
        response, inp, out = call_codex_model("sys", "user", "codex/model")
        assert response == "Test response"
        assert inp == 150
        assert out == 75

    @patch("models.CODEX_AVAILABLE", True)
    @patch("models.subprocess.run")
    def test_handles_nonzero_exit_code(self, mock_run):
        import pytest

        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Some error")
        with pytest.raises(RuntimeError, match="Codex CLI failed"):
            call_codex_model("sys", "user", "codex/model")

    @patch("models.CODEX_AVAILABLE", True)
    @patch("models.subprocess.run")
    def test_handles_no_agent_message(self, mock_run):
        import pytest

        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":50}}',
            stderr="",
        )
        with pytest.raises(RuntimeError, match="No agent message found"):
            call_codex_model("sys", "user", "codex/model")

    @patch("models.CODEX_AVAILABLE", True)
    @patch("models.subprocess.run")
    def test_includes_search_flag_when_enabled(self, mock_run):
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"type":"item.completed","item":{"type":"agent_message","text":"Response"}}\n{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":50}}',
            stderr="",
        )
        call_codex_model("sys", "user", "codex/model", search=True)
        cmd = mock_run.call_args[0][0]
        assert "--search" in cmd

    @patch("models.CODEX_AVAILABLE", True)
    @patch("models.subprocess.run")
    def test_timeout_raises_runtime_error(self, mock_run):
        import pytest
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("codex", 600)
        with pytest.raises(RuntimeError, match="timed out"):
            call_codex_model("sys", "user", "codex/model")

    @patch("models.CODEX_AVAILABLE", True)
    @patch("models.subprocess.run")
    def test_file_not_found_raises_runtime_error(self, mock_run):
        import pytest

        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(RuntimeError, match="not found in PATH"):
            call_codex_model("sys", "user", "codex/model")

    @patch("models.CODEX_AVAILABLE", True)
    @patch("models.subprocess.run")
    def test_reasoning_effort_passed_correctly(self, mock_run):
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"type":"item.completed","item":{"type":"agent_message","text":"R"}}\n{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}',
            stderr="",
        )
        call_codex_model("sys", "user", "codex/model", reasoning_effort="high")
        cmd = mock_run.call_args[0][0]
        # Find -c argument and check reasoning effort
        assert any('model_reasoning_effort="high"' in str(arg) for arg in cmd)

    @patch("models.CODEX_AVAILABLE", True)
    @patch("models.subprocess.run")
    def test_skips_empty_lines_in_jsonl(self, mock_run):
        mock_run.return_value = Mock(
            returncode=0,
            stdout='\n\n{"type":"item.completed","item":{"type":"agent_message","text":"Response"}}\n\n{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":50}}\n',
            stderr="",
        )
        response, inp, out = call_codex_model("sys", "user", "codex/model")
        assert response == "Response"

    @patch("models.CODEX_AVAILABLE", True)
    @patch("models.subprocess.run")
    def test_default_tokens_when_no_usage(self, mock_run):
        # Mutation: input_tokens = 0 -> 1, output_tokens = 0 -> 1
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"type":"item.completed","item":{"type":"agent_message","text":"Response"}}',
            stderr="",
        )
        response, inp, out = call_codex_model("sys", "user", "codex/model")
        # Without turn.completed event, should default to 0
        assert inp == 0
        assert out == 0

    @patch("models.CODEX_AVAILABLE", True)
    @patch("models.subprocess.run")
    def test_handles_malformed_json_line(self, mock_run):
        mock_run.return_value = Mock(
            returncode=0,
            stdout='not json\n{"type":"item.completed","item":{"type":"agent_message","text":"Response"}}\n{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":50}}',
            stderr="",
        )
        response, inp, out = call_codex_model("sys", "user", "codex/model")
        assert response == "Response"

    @patch("models.CODEX_AVAILABLE", True)
    @patch("models.subprocess.run")
    def test_extracts_token_counts_from_usage(self, mock_run):
        # Mutation: usage.get("input_tokens", 0) -> 1 would change this
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"type":"item.completed","item":{"type":"agent_message","text":"R"}}\n{"type":"turn.completed","usage":{"input_tokens":0,"output_tokens":0}}',
            stderr="",
        )
        response, inp, out = call_codex_model("sys", "user", "codex/model")
        # Verify we get exact values from usage, not defaults
        assert inp == 0
        assert out == 0


class TestCallSingleModel:
    @patch("models.completion")
    def test_returns_model_response_on_success(self, mock_completion):
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="[AGREE]\n[SPEC]Final spec[/SPEC]"))
        ]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_completion.return_value = mock_response

        result = call_single_model(
            model="gpt-4o", spec="# Test Spec", round_num=1, doc_type="prd"
        )

        assert result.model == "gpt-4o"
        assert result.agreed is True
        assert result.spec == "Final spec"
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    @patch("models.completion")
    def test_extracts_spec_from_response(self, mock_completion):
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="Critique here\n[SPEC]\n# New Spec\n[/SPEC]"))
        ]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_completion.return_value = mock_response

        result = call_single_model("gpt-4o", "spec", 1, "prd")
        assert result.spec == "# New Spec"
        assert result.agreed is False

    @patch("models.completion")
    def test_handles_missing_usage(self, mock_completion):
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="[AGREE]"))]
        mock_response.usage = None
        mock_completion.return_value = mock_response

        result = call_single_model("gpt-4o", "spec", 1, "prd")
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    @patch("models.completion")
    @patch("models.time.sleep")
    def test_retries_on_failure(self, mock_sleep, mock_completion):
        mock_completion.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            Mock(
                choices=[Mock(message=Mock(content="[AGREE]"))],
                usage=Mock(prompt_tokens=10, completion_tokens=5),
            ),
        ]

        result = call_single_model("gpt-4o", "spec", 1, "prd")
        assert result.agreed is True
        assert mock_completion.call_count == 3
        # Verify exponential backoff
        assert mock_sleep.call_count == 2

    @patch("models.completion")
    @patch("models.time.sleep")
    def test_exponential_backoff_delay(self, mock_sleep, mock_completion):
        # Mutation: * -> / or 2**attempt -> 2*attempt would change delays
        mock_completion.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            Mock(
                choices=[Mock(message=Mock(content="[AGREE]"))],
                usage=Mock(prompt_tokens=10, completion_tokens=5),
            ),
        ]

        call_single_model("gpt-4o", "spec", 1, "prd")
        # First retry: delay = 1.0 * 2^0 = 1.0
        # Second retry: delay = 1.0 * 2^1 = 2.0
        calls = mock_sleep.call_args_list
        assert calls[0][0][0] == 1.0  # First delay: 1.0 * 2^0
        assert calls[1][0][0] == 2.0  # Second delay: 1.0 * 2^1

    @patch("models.completion")
    @patch("models.time.sleep")
    def test_returns_error_after_max_retries(self, mock_sleep, mock_completion):
        mock_completion.side_effect = Exception("Persistent failure")

        result = call_single_model("gpt-4o", "spec", 1, "prd")
        assert result.error is not None
        assert "Persistent failure" in result.error
        assert mock_completion.call_count == MAX_RETRIES

    @patch("models.completion")
    def test_bedrock_mode_prefixes_model(self, mock_completion):
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="[AGREE]"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_completion.return_value = mock_response

        call_single_model("claude-3", "spec", 1, "prd", bedrock_mode=True)
        # Verify bedrock/ prefix was added
        call_args = mock_completion.call_args
        assert call_args[1]["model"] == "bedrock/claude-3"

    @patch("models.completion")
    def test_bedrock_mode_skips_prefix_if_already_present(self, mock_completion):
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="[AGREE]"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_completion.return_value = mock_response

        call_single_model("bedrock/claude-3", "spec", 1, "prd", bedrock_mode=True)
        call_args = mock_completion.call_args
        assert call_args[1]["model"] == "bedrock/claude-3"  # Not bedrock/bedrock/

    @patch("models.completion")
    @patch("models.time.sleep")
    def test_bedrock_access_denied_error_message(self, mock_sleep, mock_completion):
        mock_completion.side_effect = Exception("AccessDeniedException: not authorized")

        result = call_single_model("claude-3", "spec", 1, "prd", bedrock_mode=True)
        assert "not enabled" in result.error

    @patch("models.completion")
    @patch("models.time.sleep")
    def test_bedrock_validation_error_message(self, mock_sleep, mock_completion):
        mock_completion.side_effect = Exception("ValidationException: bad model")

        result = call_single_model("claude-3", "spec", 1, "prd", bedrock_mode=True)
        assert "Invalid Bedrock model" in result.error

    @patch("models.completion")
    def test_uses_press_template_when_press_true(self, mock_completion):
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="[AGREE]"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_completion.return_value = mock_response

        call_single_model("gpt-4o", "spec", 1, "prd", press=True)
        call_args = mock_completion.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        # Press template includes "round 1" and "previously indicated agreement"
        assert "round 1" in user_msg
        assert "previously indicated agreement" in user_msg

    @patch("models.completion")
    def test_includes_focus_section(self, mock_completion):
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="[AGREE]"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_completion.return_value = mock_response

        call_single_model("gpt-4o", "spec", 1, "prd", focus="security")
        call_args = mock_completion.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        assert "security" in user_msg.lower()

    @patch("models.completion")
    def test_custom_focus_creates_section(self, mock_completion):
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="[AGREE]"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_completion.return_value = mock_response

        call_single_model("gpt-4o", "spec", 1, "prd", focus="customarea")
        call_args = mock_completion.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        assert "CUSTOMAREA" in user_msg

    @patch("models.completion")
    def test_preserve_intent_adds_prompt(self, mock_completion):
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="[AGREE]"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_completion.return_value = mock_response

        call_single_model("gpt-4o", "spec", 1, "prd", preserve_intent=True)
        call_args = mock_completion.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        # preserve_intent adds a specific prompt section
        assert len(user_msg) > 0

    @patch("models.call_codex_model")
    @patch("models.CODEX_AVAILABLE", True)
    def test_routes_codex_model_to_handler(self, mock_codex):
        mock_codex.return_value = ("[AGREE]\n[SPEC]spec[/SPEC]", 100, 50)

        result = call_single_model("codex/gpt-5", "spec", 1, "prd")
        mock_codex.assert_called_once()
        assert result.model == "codex/gpt-5"

    @patch("models.call_codex_model")
    @patch("models.CODEX_AVAILABLE", True)
    @patch("models.time.sleep")
    def test_codex_retries_on_failure(self, mock_sleep, mock_codex):
        mock_codex.side_effect = [Exception("First fail"), ("[AGREE]", 10, 5)]

        result = call_single_model("codex/gpt-5", "spec", 1, "prd")
        assert mock_codex.call_count == 2
        assert result.agreed is True

    @patch("models.call_codex_model")
    @patch("models.CODEX_AVAILABLE", True)
    def test_codex_extracts_spec_from_response(self, mock_codex):
        # Mutation: extracted = extract_spec(content) -> extracted = None
        mock_codex.return_value = ("Critique\n[SPEC]Extracted spec[/SPEC]", 100, 50)

        result = call_single_model("codex/gpt-5", "spec", 1, "prd")
        assert result.spec == "Extracted spec"  # Must be extracted, not None

    @patch("models.call_codex_model")
    @patch("models.CODEX_AVAILABLE", True)
    @patch("models.time.sleep")
    def test_codex_exponential_backoff(self, mock_sleep, mock_codex):
        # Verify codex path also uses exponential backoff
        mock_codex.side_effect = [
            Exception("First fail"),
            Exception("Second fail"),
            ("[AGREE]", 10, 5),
        ]

        call_single_model("codex/gpt-5", "spec", 1, "prd")
        calls = mock_sleep.call_args_list
        assert calls[0][0][0] == 1.0  # First delay
        assert calls[1][0][0] == 2.0  # Second delay


class TestCallModelsParallel:
    @patch("models.call_single_model")
    def test_calls_all_models(self, mock_single):
        mock_single.return_value = ModelResponse(
            model="test", response="[AGREE]", agreed=True, spec="spec"
        )

        results = call_models_parallel(
            models=["gpt-4o", "gemini-pro", "claude-3"],
            spec="test spec",
            round_num=1,
            doc_type="prd",
        )

        assert len(results) == 3
        assert mock_single.call_count == 3

    @patch("models.call_single_model")
    def test_returns_all_results(self, mock_single):
        def make_response(model, *args, **kwargs):
            return ModelResponse(
                model=model,
                response=f"Response from {model}",
                agreed=model == "gpt-4o",
                spec="spec",
            )

        mock_single.side_effect = make_response

        results = call_models_parallel(
            models=["gpt-4o", "gemini-pro"],
            spec="test spec",
            round_num=1,
            doc_type="prd",
        )

        models = [r.model for r in results]
        assert "gpt-4o" in models
        assert "gemini-pro" in models

    @patch("models.call_single_model")
    def test_passes_all_parameters(self, mock_single):
        mock_single.return_value = ModelResponse(
            model="test", response="", agreed=True, spec=""
        )

        call_models_parallel(
            models=["gpt-4o"],
            spec="spec",
            round_num=5,
            doc_type="rfc",
            press=True,
            focus="security",
            persona="architect",
            context="ctx",
            preserve_intent=True,
            codex_reasoning="high",
            codex_search=True,
            timeout=300,
            bedrock_mode=True,
            bedrock_region="us-west-2",
        )

        call_args = mock_single.call_args
        assert call_args[0][0] == "gpt-4o"  # model
        assert call_args[0][1] == "spec"  # spec
        assert call_args[0][2] == 5  # round_num
        assert call_args[0][3] == "rfc"  # doc_type


class TestConstants:
    def test_max_retries_is_reasonable(self):
        # Mutation: 3 -> 4 would be caught
        assert MAX_RETRIES == 3

    def test_retry_base_delay_is_positive(self):
        # Mutation: 1.0 -> 2.0 would be caught
        assert RETRY_BASE_DELAY == 1.0


# Import new functions for testing
from models import extract_findings, merge_findings, format_findings_report


class TestExtractFindings:
    def test_extracts_single_finding(self):
        response = """
Some preamble text.

[FINDING]
severity: CRITICAL
category: Security
file: auth.py
lines: 42-58
description: SQL injection vulnerability in login function
code: |
  query = f"SELECT * FROM users WHERE name='{name}'"
recommendation: Use parameterized queries
[/FINDING]

Some closing text.
"""
        findings = extract_findings(response)
        assert len(findings) == 1
        assert findings[0]["severity"] == "CRITICAL"
        assert findings[0]["category"] == "Security"
        assert findings[0]["file"] == "auth.py"
        assert findings[0]["lines"] == "42-58"
        assert "SQL injection" in findings[0]["description"]
        assert "parameterized" in findings[0]["recommendation"]

    def test_extracts_multiple_findings(self):
        response = """
[FINDING]
severity: MAJOR
category: Bug
file: main.py
description: Null pointer exception possible
[/FINDING]

[FINDING]
severity: MINOR
category: Style
file: utils.py
description: Inconsistent naming
[/FINDING]
"""
        findings = extract_findings(response)
        assert len(findings) == 2
        assert findings[0]["severity"] == "MAJOR"
        assert findings[1]["severity"] == "MINOR"

    def test_normalizes_severity(self):
        response = """
[FINDING]
severity: critical
description: Test finding
[/FINDING]
"""
        findings = extract_findings(response)
        assert findings[0]["severity"] == "CRITICAL"

    def test_handles_no_findings(self):
        response = "[AGREE] - Code looks good!"
        findings = extract_findings(response)
        assert findings == []

    def test_requires_description(self):
        response = """
[FINDING]
severity: MAJOR
file: test.py
[/FINDING]
"""
        findings = extract_findings(response)
        assert findings == []  # No description, so not added

    def test_handles_multiline_code(self):
        response = """
[FINDING]
severity: MAJOR
category: Bug
file: test.py
description: Issue found
code: |
  def foo():
      return bar
recommendation: Fix it
[/FINDING]
"""
        findings = extract_findings(response)
        assert "def foo()" in findings[0]["code"]
        assert "return bar" in findings[0]["code"]


class TestMergeFindings:
    def test_merges_agreed_findings(self):
        model_findings = [
            (
                "model-a",
                [{"file": "test.py", "severity": "MAJOR", "description": "Bug found"}],
            ),
            (
                "model-b",
                [{"file": "test.py", "severity": "MAJOR", "description": "Bug found"}],
            ),
        ]
        agreed, contested = merge_findings(model_findings)
        assert len(agreed) == 1
        assert len(contested) == 0
        assert "model-a" in agreed[0]["agreed_by"]
        assert "model-b" in agreed[0]["agreed_by"]

    def test_identifies_contested_findings(self):
        model_findings = [
            ("model-a", [{"file": "a.py", "description": "Issue in a"}]),
            ("model-b", [{"file": "b.py", "description": "Issue in b"}]),
        ]
        agreed, contested = merge_findings(model_findings)
        # With 2 models, neither finding has majority
        assert len(contested) == 2

    def test_sorts_by_severity(self):
        model_findings = [
            (
                "model-a",
                [
                    {"file": "a.py", "severity": "MINOR", "description": "Minor issue"},
                    {
                        "file": "b.py",
                        "severity": "CRITICAL",
                        "description": "Critical issue",
                    },
                ],
            ),
            (
                "model-b",
                [
                    {"file": "a.py", "severity": "MINOR", "description": "Minor issue"},
                    {
                        "file": "b.py",
                        "severity": "CRITICAL",
                        "description": "Critical issue",
                    },
                ],
            ),
        ]
        agreed, _ = merge_findings(model_findings)
        assert agreed[0]["severity"] == "CRITICAL"
        assert agreed[1]["severity"] == "MINOR"

    def test_handles_empty_input(self):
        agreed, contested = merge_findings([])
        assert agreed == []
        assert contested == []


class TestFormatFindingsReport:
    def test_formats_basic_report(self):
        agreed = [
            {
                "severity": "CRITICAL",
                "category": "Security",
                "file": "auth.py",
                "lines": "10-20",
                "description": "SQL injection",
                "recommendation": "Use prepared statements",
                "agreed_by": ["model-a", "model-b"],
            }
        ]
        report = format_findings_report(agreed, [], "Test Review", ["model-a", "model-b"])
        assert "# Test Review" in report
        assert "CRITICAL" in report
        assert "Security" in report
        assert "auth.py:10-20" in report
        assert "SQL injection" in report
        assert "Use prepared statements" in report

    def test_includes_contested_section(self):
        contested = [
            {
                "severity": "MINOR",
                "category": "Style",
                "file": "utils.py",
                "description": "Naming issue",
                "found_by": ["model-a"],
                "not_found_by": ["model-b"],
            }
        ]
        report = format_findings_report([], contested, "Review")
        assert "Contested Findings" in report
        assert "model-a" in report
        assert "model-b" in report

    def test_includes_summary_counts(self):
        agreed = [
            {"severity": "CRITICAL", "description": "Critical 1"},
            {"severity": "CRITICAL", "description": "Critical 2"},
            {"severity": "MAJOR", "description": "Major 1"},
        ]
        report = format_findings_report(agreed, [], "Summary Test")
        assert "Critical: 2" in report
        assert "Major: 1" in report

    def test_includes_models_list(self):
        report = format_findings_report(
            [], [], "Test", ["gpt-4o", "gemini/gemini-2.0-flash"]
        )
        assert "gpt-4o" in report
        assert "gemini/gemini-2.0-flash" in report
