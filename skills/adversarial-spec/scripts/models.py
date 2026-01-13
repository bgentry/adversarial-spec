"""Model calling, cost tracking, and response handling."""

import os
import sys
import json
import time
import subprocess
import difflib
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

os.environ["LITELLM_LOG"] = "ERROR"

try:
    import litellm
    from litellm import completion

    litellm.suppress_debug_info = True
except ImportError:
    print(
        "Error: litellm package not installed. Run: pip install litellm",
        file=sys.stderr,
    )
    sys.exit(1)

from prompts import (
    FOCUS_AREAS,
    PRESERVE_INTENT_PROMPT,
    get_system_prompt,
    get_doc_type_name,
    get_focus_areas,
    get_review_prompt_template,
)
from providers import (
    MODEL_COSTS,
    DEFAULT_COST,
    CODEX_AVAILABLE,
    DEFAULT_CODEX_REASONING,
)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds


@dataclass
class ModelResponse:
    """Response from a model critique."""

    model: str
    response: str
    agreed: bool
    spec: Optional[str]
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class CostTracker:
    """Track token usage and costs across model calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Add usage for a model call and return the cost."""
        costs = MODEL_COSTS.get(model, DEFAULT_COST)
        cost = (input_tokens / 1_000_000 * costs["input"]) + (
            output_tokens / 1_000_000 * costs["output"]
        )

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        if model not in self.by_model:
            self.by_model[model] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        self.by_model[model]["input_tokens"] += input_tokens
        self.by_model[model]["output_tokens"] += output_tokens
        self.by_model[model]["cost"] += cost

        return cost

    def summary(self) -> str:
        """Generate cost summary string."""
        lines = ["", "=== Cost Summary ==="]
        lines.append(
            f"Total tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out"
        )
        lines.append(f"Total cost: ${self.total_cost:.4f}")
        if len(self.by_model) > 1:
            lines.append("")
            lines.append("By model:")
            for model, data in self.by_model.items():
                lines.append(
                    f"  {model}: ${data['cost']:.4f} ({data['input_tokens']:,} in / {data['output_tokens']:,} out)"
                )
        return "\n".join(lines)


# Global cost tracker instance
cost_tracker = CostTracker()


def load_context_files(context_paths: list[str]) -> str:
    """Load and format context files for inclusion in prompts."""
    if not context_paths:
        return ""

    sections = []
    for path in context_paths:
        try:
            content = Path(path).read_text()
            sections.append(f"### Context: {path}\n```\n{content}\n```")
        except Exception as e:
            sections.append(f"### Context: {path}\n[Error loading file: {e}]")

    return (
        "## Additional Context\nThe following documents are provided as context:\n\n"
        + "\n\n".join(sections)
    )


def detect_agreement(response: str) -> bool:
    """Check if response indicates agreement."""
    return "[AGREE]" in response


def extract_spec(response: str) -> Optional[str]:
    """Extract spec content from [SPEC]...[/SPEC] tags."""
    if "[SPEC]" not in response or "[/SPEC]" not in response:
        return None
    start = response.find("[SPEC]") + len("[SPEC]")
    end = response.find("[/SPEC]")
    return response[start:end].strip()


def extract_tasks(response: str) -> list[dict]:
    """Extract tasks from export-tasks response."""
    tasks = []
    parts = response.split("[TASK]")
    for part in parts[1:]:
        if "[/TASK]" not in part:
            continue
        task_text = part.split("[/TASK]")[0].strip()
        task: dict[str, str | list[str]] = {}
        current_key: str | None = None
        current_value: list[str] = []

        for line in task_text.split("\n"):
            line = line.strip()
            if line.startswith("title:"):
                if current_key:
                    task[current_key] = (
                        "\n".join(current_value).strip()
                        if len(current_value) > 1
                        else current_value[0]
                        if current_value
                        else ""
                    )
                current_key = "title"
                current_value = [line[6:].strip()]
            elif line.startswith("type:"):
                if current_key:
                    task[current_key] = (
                        "\n".join(current_value).strip()
                        if len(current_value) > 1
                        else current_value[0]
                        if current_value
                        else ""
                    )
                current_key = "type"
                current_value = [line[5:].strip()]
            elif line.startswith("priority:"):
                if current_key:
                    task[current_key] = (
                        "\n".join(current_value).strip()
                        if len(current_value) > 1
                        else current_value[0]
                        if current_value
                        else ""
                    )
                current_key = "priority"
                current_value = [line[9:].strip()]
            elif line.startswith("description:"):
                if current_key:
                    task[current_key] = (
                        "\n".join(current_value).strip()
                        if len(current_value) > 1
                        else current_value[0]
                        if current_value
                        else ""
                    )
                current_key = "description"
                current_value = [line[12:].strip()]
            elif line.startswith("acceptance_criteria:"):
                if current_key:
                    task[current_key] = (
                        "\n".join(current_value).strip()
                        if len(current_value) > 1
                        else current_value[0]
                        if current_value
                        else ""
                    )
                current_key = "acceptance_criteria"
                current_value = []
            elif line.startswith("- ") and current_key == "acceptance_criteria":
                current_value.append(line[2:])
            elif current_key:
                current_value.append(line)

        if current_key:
            task[current_key] = (
                current_value
                if current_key == "acceptance_criteria"
                else "\n".join(current_value).strip()
            )

        if task.get("title"):
            tasks.append(task)

    return tasks


def extract_findings(response: str) -> list[dict]:
    """Extract code review findings from [FINDING]...[/FINDING] blocks.

    Args:
        response: Model response containing [FINDING] blocks.

    Returns:
        List of finding dictionaries with severity, category, file, etc.
    """
    findings = []
    parts = response.split("[FINDING]")

    for part in parts[1:]:
        if "[/FINDING]" not in part:
            continue
        finding_text = part.split("[/FINDING]")[0].strip()
        finding: dict[str, str] = {}
        current_key: str | None = None
        current_value: list[str] = []

        in_code_block = False
        for line in finding_text.split("\n"):
            stripped = line.strip()

            # If in code block, only exit when we see an unindented known key
            if in_code_block:
                # Check if this is an unindented line starting with a known key
                is_new_key = False
                if line and not line[0].isspace():
                    for key in ["severity", "category", "file", "lines", "description", "code", "recommendation"]:
                        if stripped.lower().startswith(f"{key}:"):
                            is_new_key = True
                            in_code_block = False
                            break
                if not is_new_key:
                    current_value.append(line.rstrip())
                    continue

            # Check for known keys
            for key in [
                "severity",
                "category",
                "file",
                "lines",
                "description",
                "code",
                "recommendation",
            ]:
                if stripped.lower().startswith(f"{key}:"):
                    # Save previous key's value
                    if current_key:
                        finding[current_key] = "\n".join(current_value).strip()
                    current_key = key
                    # Handle value after colon
                    value_after_colon = stripped[len(key) + 1 :].strip()
                    # For 'code' field, check if it starts with '|' for multiline
                    if key == "code" and value_after_colon == "|":
                        current_value = []
                        in_code_block = True
                    else:
                        current_value = [value_after_colon] if value_after_colon else []
                    break
            else:
                # No key matched, append to current value
                if current_key:
                    current_value.append(line.rstrip())

        # Save the last key's value
        if current_key:
            finding[current_key] = "\n".join(current_value).strip()

        # Normalize severity
        if "severity" in finding:
            finding["severity"] = finding["severity"].upper()
            # Handle variations like "CRITICAL:" or "critical"
            for sev in ["CRITICAL", "MAJOR", "MINOR", "NITPICK"]:
                if sev in finding["severity"]:
                    finding["severity"] = sev
                    break

        # Only add if we have at least description
        if finding.get("description"):
            findings.append(finding)

    return findings


def merge_findings(
    all_model_findings: list[tuple[str, list[dict]]]
) -> tuple[list[dict], list[dict]]:
    """Merge findings from multiple models, tracking agreement.

    Args:
        all_model_findings: List of (model_name, findings_list) tuples.

    Returns:
        Tuple of (agreed_findings, contested_findings).
        Each finding has 'agreed_by' or 'contested_by' field added.
    """
    if not all_model_findings:
        return [], []

    # Group findings by file + severity + description prefix
    # Including severity ensures we don't merge CRITICAL with MINOR issues
    def finding_key(f: dict) -> str:
        file_part = f.get("file", "unknown")[:50]
        severity_part = f.get("severity", "UNKNOWN").upper()
        desc_part = f.get("description", "")[:50].lower()
        return f"{file_part}:{severity_part}:{desc_part}"

    # Collect all findings with model attribution
    finding_groups: dict[str, list[tuple[str, dict]]] = {}
    for model_name, findings in all_model_findings:
        for finding in findings:
            key = finding_key(finding)
            if key not in finding_groups:
                finding_groups[key] = []
            finding_groups[key].append((model_name, finding))

    agreed = []
    contested = []
    total_models = len(all_model_findings)

    for key, model_findings in finding_groups.items():
        models_found = [m for m, _ in model_findings]
        # Use the most detailed finding (longest description)
        best_finding = max(
            [f for _, f in model_findings],
            key=lambda f: len(f.get("description", "")),
        )

        if len(models_found) > total_models / 2:  # Strict majority agreement
            best_finding["agreed_by"] = models_found
            agreed.append(best_finding)
        else:
            best_finding["found_by"] = models_found
            best_finding["not_found_by"] = [
                m for m, _ in all_model_findings if m not in models_found
            ]
            contested.append(best_finding)

    # Sort by severity
    severity_order = {"CRITICAL": 0, "MAJOR": 1, "MINOR": 2, "NITPICK": 3}
    agreed.sort(key=lambda f: severity_order.get(f.get("severity", "MINOR"), 2))
    contested.sort(key=lambda f: severity_order.get(f.get("severity", "MINOR"), 2))

    return agreed, contested


def format_findings_report(
    agreed: list[dict],
    contested: list[dict],
    title: str = "Code Review",
    models: list[str] | None = None,
) -> str:
    """Format findings into a markdown report.

    Args:
        agreed: List of agreed-upon findings.
        contested: List of contested findings.
        title: Report title.
        models: List of model names that participated.

    Returns:
        Formatted markdown report.
    """
    # Count by severity
    severity_counts = {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0, "NITPICK": 0}
    for f in agreed:
        sev = f.get("severity", "MINOR")
        if sev in severity_counts:
            severity_counts[sev] += 1

    report = f"""# {title}

## Summary
- Total findings: {len(agreed)} agreed, {len(contested)} contested
- Critical: {severity_counts['CRITICAL']}
- Major: {severity_counts['MAJOR']}
- Minor: {severity_counts['MINOR']}
- Nitpicks: {severity_counts['NITPICK']}
"""
    if models:
        report += f"- Models: {', '.join(models)}\n"

    if agreed:
        report += "\n## Agreed Findings\n\n"
        for i, f in enumerate(agreed, 1):
            sev = f.get("severity", "UNKNOWN")
            cat = f.get("category", "General")
            file_loc = f.get("file", "unknown")
            lines = f.get("lines", "")
            if lines:
                file_loc = f"{file_loc}:{lines}"

            report += f"### {i}. [{sev}] {cat}\n\n"
            report += f"**Location:** `{file_loc}`\n\n"
            report += f"**Description:** {f.get('description', 'No description')}\n\n"

            if f.get("code"):
                report += f"**Code:**\n```\n{f['code']}\n```\n\n"

            if f.get("recommendation"):
                report += f"**Recommendation:** {f['recommendation']}\n\n"

            if f.get("agreed_by"):
                report += f"*Found by: {', '.join(f['agreed_by'])}*\n\n"

            report += "---\n\n"

    if contested:
        report += "\n## Contested Findings\n\n"
        report += "*These findings were not agreed upon by all models.*\n\n"
        for i, f in enumerate(contested, 1):
            sev = f.get("severity", "UNKNOWN")
            cat = f.get("category", "General")
            file_loc = f.get("file", "unknown")

            report += f"### {i}. [{sev}] {cat}\n\n"
            report += f"**Location:** `{file_loc}`\n\n"
            report += f"**Description:** {f.get('description', 'No description')}\n\n"

            if f.get("found_by"):
                report += f"*Found by: {', '.join(f['found_by'])}*\n"
            if f.get("not_found_by"):
                report += f"*Not flagged by: {', '.join(f['not_found_by'])}*\n\n"

            report += "---\n\n"

    return report


def get_critique_summary(response: str, max_length: int = 300) -> str:
    """Get a summary of the critique portion of a response."""
    spec_start = response.find("[SPEC]")
    if spec_start > 0:
        critique = response[:spec_start].strip()
    else:
        critique = response

    if len(critique) > max_length:
        critique = critique[:max_length] + "..."
    return critique


def generate_diff(previous: str, current: str) -> str:
    """Generate unified diff between two specs."""
    prev_lines = previous.splitlines(keepends=True)
    curr_lines = current.splitlines(keepends=True)

    diff = difflib.unified_diff(
        prev_lines, curr_lines, fromfile="previous", tofile="current", lineterm=""
    )
    return "".join(diff)


def call_codex_model(
    system_prompt: str,
    user_message: str,
    model: str,
    reasoning_effort: str = DEFAULT_CODEX_REASONING,
    timeout: int = 600,
    search: bool = False,
) -> tuple[str, int, int]:
    """
    Call Codex CLI in headless mode using ChatGPT subscription.

    Args:
        system_prompt: System instructions for the model
        user_message: User prompt to send
        model: Model name (e.g., "codex/gpt-5.2-codex" -> uses "gpt-5.2-codex")
        reasoning_effort: Thinking level (minimal, low, medium, high, xhigh). Default: xhigh
        timeout: Timeout in seconds (default 10 minutes)
        search: Enable web search capability for Codex

    Returns:
        Tuple of (response_text, input_tokens, output_tokens)

    Raises:
        RuntimeError: If Codex CLI is not available or fails
    """
    if not CODEX_AVAILABLE:
        raise RuntimeError(
            "Codex CLI not found. Install with: npm install -g @openai/codex"
        )

    # Extract actual model name from "codex/model" format
    actual_model = model.split("/", 1)[1] if "/" in model else model

    # Combine system prompt and user message for Codex
    full_prompt = f"""SYSTEM INSTRUCTIONS:
{system_prompt}

USER REQUEST:
{user_message}"""

    try:
        cmd = [
            "codex",
            "exec",
            "--json",
            "--full-auto",
            "--model",
            actual_model,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
        ]
        if search:
            cmd.append("--search")
        cmd.append(full_prompt)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            error_msg = (
                result.stderr.strip() or f"Codex exited with code {result.returncode}"
            )
            raise RuntimeError(f"Codex CLI failed: {error_msg}")

        # Parse JSONL output to extract agent messages
        response_text = ""
        input_tokens = 0
        output_tokens = 0

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)

                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        response_text = item.get("text", "")

                if event.get("type") == "turn.completed":
                    usage = event.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)

            except json.JSONDecodeError:
                continue

        if not response_text:
            raise RuntimeError("No agent message found in Codex output")

        return response_text, input_tokens, output_tokens

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Codex CLI timed out after {timeout}s")
    except FileNotFoundError:
        raise RuntimeError("Codex CLI not found in PATH")


def call_single_model(
    model: str,
    spec: str,
    round_num: int,
    doc_type: str,
    press: bool = False,
    focus: Optional[str] = None,
    persona: Optional[str] = None,
    context: Optional[str] = None,
    preserve_intent: bool = False,
    codex_reasoning: str = DEFAULT_CODEX_REASONING,
    codex_search: bool = False,
    timeout: int = 600,
    bedrock_mode: bool = False,
    bedrock_region: Optional[str] = None,
) -> ModelResponse:
    """Send spec to a single model and return response with retry on failure."""
    # Handle Bedrock routing
    actual_model = model
    if bedrock_mode:
        if bedrock_region:
            os.environ["AWS_REGION"] = bedrock_region
        if not model.startswith("bedrock/"):
            actual_model = f"bedrock/{model}"

    system_prompt = get_system_prompt(doc_type, persona)
    doc_type_name = get_doc_type_name(doc_type)

    # Get appropriate focus areas for document type
    available_focus_areas = get_focus_areas(doc_type)
    focus_section = ""
    if focus and focus.lower() in available_focus_areas:
        focus_section = available_focus_areas[focus.lower()]
    elif focus and focus.lower() in FOCUS_AREAS:
        # Fall back to generic focus areas
        focus_section = FOCUS_AREAS[focus.lower()]
    elif focus:
        focus_section = f"**CRITICAL FOCUS: {focus.upper()}**\nPrioritize analysis of {focus} concerns above all else."

    if preserve_intent:
        focus_section = PRESERVE_INTENT_PROMPT + "\n\n" + focus_section

    context_section = context if context else ""

    # Get appropriate prompt template for document type
    template = get_review_prompt_template(doc_type, press)
    user_message = template.format(
        round=round_num,
        doc_type_name=doc_type_name,
        spec=spec,
        focus_section=focus_section,
        context_section=context_section,
    )

    # Route Codex CLI models to dedicated handler
    if model.startswith("codex/"):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                content, input_tokens, output_tokens = call_codex_model(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    model=model,
                    reasoning_effort=codex_reasoning,
                    timeout=timeout,
                    search=codex_search,
                )
                agreed = "[AGREE]" in content
                extracted = extract_spec(content)

                if not agreed and not extracted:
                    print(
                        f"Warning: {model} provided critique but no [SPEC] tags found. Response may be malformed.",
                        file=sys.stderr,
                    )

                cost = cost_tracker.add(model, input_tokens, output_tokens)

                return ModelResponse(
                    model=model,
                    response=content,
                    agreed=agreed,
                    spec=extracted,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                )
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2**attempt)
                    print(
                        f"Warning: {model} failed (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}. Retrying in {delay:.1f}s...",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                else:
                    print(
                        f"Error: {model} failed after {MAX_RETRIES} attempts: {last_error}",
                        file=sys.stderr,
                    )

        return ModelResponse(
            model=model, response="", agreed=False, spec=None, error=last_error
        )

    # Standard litellm path for all other providers
    last_error = None
    display_model = model

    for attempt in range(MAX_RETRIES):
        try:
            response = completion(
                model=actual_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.7,
                max_tokens=8000,
                timeout=timeout,
            )
            content = response.choices[0].message.content
            agreed = "[AGREE]" in content
            extracted = extract_spec(content)

            if not agreed and not extracted:
                print(
                    f"Warning: {display_model} provided critique but no [SPEC] tags found. Response may be malformed.",
                    file=sys.stderr,
                )

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            cost = cost_tracker.add(display_model, input_tokens, output_tokens)

            return ModelResponse(
                model=display_model,
                response=content,
                agreed=agreed,
                spec=extracted,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
            )
        except Exception as e:
            last_error = str(e)
            if bedrock_mode:
                if "AccessDeniedException" in last_error:
                    last_error = (
                        f"Model not enabled in your Bedrock account: {display_model}"
                    )
                elif "ValidationException" in last_error:
                    last_error = f"Invalid Bedrock model ID: {display_model}"

            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                print(
                    f"Warning: {display_model} failed (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}. Retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                print(
                    f"Error: {display_model} failed after {MAX_RETRIES} attempts: {last_error}",
                    file=sys.stderr,
                )

    return ModelResponse(
        model=display_model, response="", agreed=False, spec=None, error=last_error
    )


def call_models_parallel(
    models: list[str],
    spec: str,
    round_num: int,
    doc_type: str,
    press: bool = False,
    focus: Optional[str] = None,
    persona: Optional[str] = None,
    context: Optional[str] = None,
    preserve_intent: bool = False,
    codex_reasoning: str = DEFAULT_CODEX_REASONING,
    codex_search: bool = False,
    timeout: int = 600,
    bedrock_mode: bool = False,
    bedrock_region: Optional[str] = None,
) -> list[ModelResponse]:
    """Call multiple models in parallel and collect responses."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
        future_to_model = {
            executor.submit(
                call_single_model,
                model,
                spec,
                round_num,
                doc_type,
                press,
                focus,
                persona,
                context,
                preserve_intent,
                codex_reasoning,
                codex_search,
                timeout,
                bedrock_mode,
                bedrock_region,
            ): model
            for model in models
        }
        for future in concurrent.futures.as_completed(future_to_model):
            results.append(future.result())
    return results
