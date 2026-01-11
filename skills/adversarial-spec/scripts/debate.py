#!/usr/bin/env python3
"""
Adversarial spec debate script.
Sends specs to multiple LLMs for critique using LiteLLM.

Usage:
    echo "spec" | python3 debate.py critique --models gpt-4o
    echo "spec" | python3 debate.py critique --models gpt-4o,gemini/gemini-2.0-flash
    echo "spec" | python3 debate.py critique --models gpt-4o,gemini/gemini-2.0-flash,xai/grok-3 --doc-type prd

Supported providers (set corresponding API key):
    OpenAI:    OPENAI_API_KEY      models: gpt-4o, gpt-4-turbo, o1, etc.
    Anthropic: ANTHROPIC_API_KEY   models: claude-sonnet-4-20250514, claude-opus-4-20250514, etc.
    Google:    GEMINI_API_KEY      models: gemini/gemini-2.0-flash, gemini/gemini-pro, etc.
    xAI:       XAI_API_KEY         models: xai/grok-3, xai/grok-beta, etc.
    Mistral:   MISTRAL_API_KEY     models: mistral/mistral-large, etc.
    Groq:      GROQ_API_KEY        models: groq/llama-3.3-70b, etc.

Document types:
    prd   - Product Requirements Document (business/product focus)
    tech  - Technical Specification / Architecture Document (engineering focus)

Exit codes:
    0 - Success
    1 - API error
    2 - Missing API key or config error
"""

import os
import sys
import argparse
import json
import concurrent.futures
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

os.environ["LITELLM_LOG"] = "ERROR"

try:
    import litellm
    from litellm import completion
    litellm.suppress_debug_info = True
except ImportError:
    print("Error: litellm package not installed. Run: pip install litellm", file=sys.stderr)
    sys.exit(1)

SYSTEM_PROMPT_PRD = """You are a senior product manager participating in adversarial spec development.

You will receive a Product Requirements Document (PRD) from another AI model. Your job is to critique it rigorously.

Analyze the PRD for:
- Clear problem definition with evidence of real user pain
- Well-defined user personas with specific, believable characteristics
- User stories in proper format (As a... I want... So that...)
- Measurable success criteria and KPIs
- Explicit scope boundaries (what's in AND out)
- Realistic risk assessment with mitigations
- Dependencies identified
- NO technical implementation details (that belongs in a tech spec)

Expected PRD structure:
- Executive Summary
- Problem Statement / Opportunity
- Target Users / Personas
- User Stories / Use Cases
- Functional Requirements
- Non-Functional Requirements
- Success Metrics / KPIs
- Scope (In/Out)
- Dependencies
- Risks and Mitigations

If you find significant issues:
- Provide a clear critique explaining each problem
- Output your revised PRD that addresses these issues
- Format: First your critique, then the revised PRD between [SPEC] and [/SPEC] tags

If the PRD is solid and ready for stakeholder review:
- Output exactly [AGREE] on its own line
- Then output the final PRD between [SPEC] and [/SPEC] tags

Be rigorous. A good PRD should let any PM or designer understand exactly what to build and why.
Push back on vague requirements, unmeasurable success criteria, and missing user context."""

SYSTEM_PROMPT_TECH = """You are a senior software architect participating in adversarial spec development.

You will receive a Technical Specification from another AI model. Your job is to critique it rigorously.

Analyze the spec for:
- Clear architectural decisions with rationale
- Complete API contracts (endpoints, methods, request/response schemas, error codes)
- Data models that handle all identified use cases
- Security threats identified and mitigated (auth, authz, input validation, data protection)
- Error scenarios enumerated with handling strategy
- Performance targets that are specific and measurable
- Deployment strategy that is repeatable and reversible
- No ambiguity an engineer would need to resolve

Expected structure:
- Overview / Context
- Goals and Non-Goals
- System Architecture
- Component Design
- API Design (full schemas, not just endpoint names)
- Data Models / Database Schema
- Infrastructure Requirements
- Security Considerations
- Error Handling Strategy
- Performance Requirements / SLAs
- Observability (logging, metrics, alerting)
- Testing Strategy
- Deployment Strategy
- Migration Plan (if applicable)
- Open Questions / Future Considerations

If you find significant issues:
- Provide a clear critique explaining each problem
- Output your revised specification that addresses these issues
- Format: First your critique, then the revised spec between [SPEC] and [/SPEC] tags

If the spec is solid and production-ready:
- Output exactly [AGREE] on its own line
- Then output the final spec between [SPEC] and [/SPEC] tags

Be rigorous. A good tech spec should let any engineer implement the system without asking clarifying questions.
Push back on incomplete APIs, missing error handling, vague performance targets, and security gaps."""

SYSTEM_PROMPT_GENERIC = """You are a senior technical reviewer participating in adversarial spec development.

You will receive a specification from another AI model. Your job:

1. Analyze the spec rigorously for:
   - Gaps in requirements
   - Ambiguous language
   - Missing edge cases
   - Security vulnerabilities
   - Scalability concerns
   - Technical feasibility issues
   - Inconsistencies between sections
   - Missing error handling
   - Unclear data models or API designs

2. If you find significant issues:
   - Provide a clear critique explaining each problem
   - Output your revised specification that addresses these issues
   - Format: First your critique, then the revised spec between [SPEC] and [/SPEC] tags

3. If the spec is solid and production-ready with no material changes needed:
   - Output exactly [AGREE] on its own line
   - Then output the final spec between [SPEC] and [/SPEC] tags

Be rigorous and demanding. Do not agree unless the spec is genuinely complete and production-ready.
Push back on weak points. The goal is convergence on an excellent spec, not quick agreement."""

REVIEW_PROMPT_TEMPLATE = """This is round {round} of adversarial spec development.

Here is the current {doc_type_name}:

{spec}

Review this document according to your criteria. Either critique and revise it, or say [AGREE] if it's production-ready."""

PRESS_PROMPT_TEMPLATE = """This is round {round} of adversarial spec development. You previously indicated agreement with this document.

Here is the current {doc_type_name}:

{spec}

**IMPORTANT: Please confirm your agreement by thoroughly reviewing the ENTIRE document.**

Before saying [AGREE], you MUST:
1. Confirm you have read every section of this document
2. List at least 3 specific sections you reviewed and what you verified in each
3. Explain WHY you agree - what makes this document complete and production-ready?
4. Identify ANY remaining concerns, however minor (even stylistic or optional improvements)

If after this thorough review you find issues you missed before, provide your critique.

If you genuinely agree after careful review, output:
1. Your verification (sections reviewed, reasons for agreement, minor concerns)
2. [AGREE] on its own line
3. The final spec between [SPEC] and [/SPEC] tags"""


@dataclass
class ModelResponse:
    model: str
    response: str
    agreed: bool
    spec: Optional[str]
    error: Optional[str] = None


def get_system_prompt(doc_type: str) -> str:
    if doc_type == "prd":
        return SYSTEM_PROMPT_PRD
    elif doc_type == "tech":
        return SYSTEM_PROMPT_TECH
    else:
        return SYSTEM_PROMPT_GENERIC


def get_doc_type_name(doc_type: str) -> str:
    if doc_type == "prd":
        return "Product Requirements Document"
    elif doc_type == "tech":
        return "Technical Specification"
    else:
        return "specification"


def call_single_model(model: str, spec: str, round_num: int, doc_type: str, press: bool = False) -> ModelResponse:
    """Send spec to a single model and return response."""
    system_prompt = get_system_prompt(doc_type)
    doc_type_name = get_doc_type_name(doc_type)

    # Use press prompt if we're pressing for confirmation
    template = PRESS_PROMPT_TEMPLATE if press else REVIEW_PROMPT_TEMPLATE
    user_message = template.format(
        round=round_num,
        doc_type_name=doc_type_name,
        spec=spec
    )

    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=8000
        )
        content = response.choices[0].message.content
        agreed = "[AGREE]" in content
        extracted = extract_spec(content)
        return ModelResponse(model=model, response=content, agreed=agreed, spec=extracted)
    except Exception as e:
        error_msg = str(e)
        return ModelResponse(model=model, response="", agreed=False, spec=None, error=error_msg)


def call_models_parallel(models: list[str], spec: str, round_num: int, doc_type: str, press: bool = False) -> list[ModelResponse]:
    """Call multiple models in parallel and collect responses."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
        future_to_model = {
            executor.submit(call_single_model, model, spec, round_num, doc_type, press): model
            for model in models
        }
        for future in concurrent.futures.as_completed(future_to_model):
            results.append(future.result())
    return results


def detect_agreement(response: str) -> bool:
    return "[AGREE]" in response


def extract_spec(response: str) -> Optional[str]:
    if "[SPEC]" not in response or "[/SPEC]" not in response:
        return None
    start = response.find("[SPEC]") + len("[SPEC]")
    end = response.find("[/SPEC]")
    return response[start:end].strip()


def get_critique_summary(response: str, max_length: int = 300) -> str:
    spec_start = response.find("[SPEC]")
    if spec_start > 0:
        critique = response[:spec_start].strip()
    else:
        critique = response

    if len(critique) > max_length:
        critique = critique[:max_length] + "..."
    return critique


def list_providers():
    providers = [
        ("OpenAI", "OPENAI_API_KEY", "gpt-5.2, gpt-4o, gpt-4-turbo, o1"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-20250514, claude-opus-4-20250514"),
        ("Google", "GEMINI_API_KEY", "gemini/gemini-2.0-flash, gemini/gemini-pro"),
        ("xAI", "XAI_API_KEY", "xai/grok-3, xai/grok-beta"),
        ("Mistral", "MISTRAL_API_KEY", "mistral/mistral-large, mistral/codestral"),
        ("Groq", "GROQ_API_KEY", "groq/llama-3.3-70b-versatile"),
        ("Together", "TOGETHER_API_KEY", "together_ai/meta-llama/Llama-3-70b"),
        ("Deepseek", "DEEPSEEK_API_KEY", "deepseek/deepseek-chat"),
    ]
    print("Supported providers:\n")
    for name, key, models in providers:
        status = "[set]" if os.environ.get(key) else "[not set]"
        print(f"  {name:12} {key:24} {status}")
        print(f"             Example models: {models}")
        print()


def send_telegram_notification(models: list[str], round_num: int, results: list[ModelResponse], poll_timeout: int) -> Optional[str]:
    """Send Telegram notification with all model responses and poll for feedback."""
    try:
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        import telegram_bot

        token, chat_id = telegram_bot.get_config()
        if not token or not chat_id:
            print("Warning: Telegram not configured. Skipping notification.", file=sys.stderr)
            return None

        # Build summary for each model
        summaries = []
        all_agreed = True
        for r in results:
            if r.error:
                summaries.append(f"`{r.model}`: ERROR - {r.error[:100]}")
                all_agreed = False
            elif r.agreed:
                summaries.append(f"`{r.model}`: AGREE")
            else:
                all_agreed = False
                summary = get_critique_summary(r.response, 200)
                summaries.append(f"`{r.model}`: {summary}")

        status = "ALL AGREE" if all_agreed else "Critiques received"
        notification = f"""*Round {round_num} complete*

Status: {status}
Models: {len(results)}

"""
        notification += "\n\n".join(summaries)

        last_update = telegram_bot.get_last_update_id(token)

        full_notification = notification + f"\n\n_Reply within {poll_timeout}s to add feedback, or wait to continue._"
        if not telegram_bot.send_long_message(token, chat_id, full_notification):
            print("Warning: Failed to send Telegram notification.", file=sys.stderr)
            return None

        feedback = telegram_bot.poll_for_reply(token, chat_id, poll_timeout, last_update)
        return feedback

    except ImportError:
        print("Warning: telegram_bot.py not found. Skipping notification.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Telegram error: {e}", file=sys.stderr)
        return None


def send_final_spec_to_telegram(spec: str, rounds: int, models: list[str], doc_type: str) -> bool:
    """Send the final converged spec to Telegram."""
    try:
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        import telegram_bot

        token, chat_id = telegram_bot.get_config()
        if not token or not chat_id:
            print("Warning: Telegram not configured. Skipping final spec notification.", file=sys.stderr)
            return False

        doc_type_name = get_doc_type_name(doc_type)
        models_str = ", ".join(f"`{m}`" for m in models)
        header = f"""*Debate complete!*

Document: {doc_type_name}
Rounds: {rounds}
Models: Claude vs {models_str}

Final document:
---"""

        if not telegram_bot.send_message(token, chat_id, header):
            return False

        return telegram_bot.send_long_message(token, chat_id, spec)

    except Exception as e:
        print(f"Warning: Failed to send final spec to Telegram: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial spec debate with multiple LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  echo "spec" | python3 debate.py critique --models gpt-4o
  echo "spec" | python3 debate.py critique --models gpt-4o,gemini/gemini-2.0-flash
  echo "spec" | python3 debate.py critique --models gpt-4o,gemini/gemini-2.0-flash,xai/grok-3 --doc-type prd
  echo "spec" | python3 debate.py critique --models gpt-4o,xai/grok-3 --telegram
  echo "spec" | python3 debate.py send-final --models gpt-4o,gemini/gemini-2.0-flash --rounds 3
  python3 debate.py providers

Document types:
  prd   - Product Requirements Document (business/product focus)
  tech  - Technical Specification / Architecture Document (engineering focus)
        """
    )
    parser.add_argument("action", choices=["critique", "providers", "send-final"],
                        help="Action to perform")
    parser.add_argument("--models", "-m", default="gpt-4o",
                        help="Comma-separated list of models (e.g., gpt-4o,gemini/gemini-2.0-flash,xai/grok-3)")
    parser.add_argument("--doc-type", "-d", choices=["prd", "tech"], default="tech",
                        help="Document type: prd or tech (default: tech)")
    parser.add_argument("--round", "-r", type=int, default=1,
                        help="Current round number")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--telegram", "-t", action="store_true",
                        help="Send Telegram notifications and poll for feedback")
    parser.add_argument("--poll-timeout", type=int, default=60,
                        help="Seconds to wait for Telegram reply (default: 60)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Total rounds completed (used with send-final)")
    parser.add_argument("--press", "-p", action="store_true",
                        help="Press models to confirm they read the full document (anti-laziness check)")
    args = parser.parse_args()

    if args.action == "providers":
        list_providers()
        return

    # Parse models list
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("Error: No models specified", file=sys.stderr)
        sys.exit(1)

    if args.action == "send-final":
        spec = sys.stdin.read().strip()
        if not spec:
            print("Error: No spec provided via stdin", file=sys.stderr)
            sys.exit(1)
        if send_final_spec_to_telegram(spec, args.rounds, models, args.doc_type):
            print("Final document sent to Telegram.")
        else:
            print("Failed to send final document to Telegram.", file=sys.stderr)
            sys.exit(1)
        return

    spec = sys.stdin.read().strip()
    if not spec:
        print("Error: No spec provided via stdin", file=sys.stderr)
        sys.exit(1)

    # Call all models in parallel
    mode = "pressing for confirmation" if args.press else "critiquing"
    print(f"Calling {len(models)} model(s) ({mode}): {', '.join(models)}...", file=sys.stderr)
    results = call_models_parallel(models, spec, args.round, args.doc_type, args.press)

    # Check for errors
    errors = [r for r in results if r.error]
    for e in errors:
        print(f"Warning: {e.model} returned error: {e.error}", file=sys.stderr)

    # Check if all agreed
    successful = [r for r in results if not r.error]
    all_agreed = all(r.agreed for r in successful) if successful else False

    # Telegram notification and feedback (if enabled)
    user_feedback = None
    if args.telegram:
        user_feedback = send_telegram_notification(models, args.round, results, args.poll_timeout)
        if user_feedback:
            print(f"Received feedback: {user_feedback}", file=sys.stderr)

    if args.json:
        output = {
            "all_agreed": all_agreed,
            "round": args.round,
            "doc_type": args.doc_type,
            "models": models,
            "results": [
                {
                    "model": r.model,
                    "agreed": r.agreed,
                    "response": r.response,
                    "spec": r.spec,
                    "error": r.error
                }
                for r in results
            ]
        }
        if user_feedback:
            output["user_feedback"] = user_feedback
        print(json.dumps(output, indent=2))
    else:
        doc_type_name = get_doc_type_name(args.doc_type)
        print(f"\n=== Round {args.round} Results ({doc_type_name}) ===\n")

        for r in results:
            print(f"--- {r.model} ---")
            if r.error:
                print(f"ERROR: {r.error}")
            elif r.agreed:
                print("[AGREE]")
            else:
                print(r.response)
            print()

        if all_agreed:
            print("=== ALL MODELS AGREE ===")
        else:
            agreed_models = [r.model for r in successful if r.agreed]
            disagreed_models = [r.model for r in successful if not r.agreed]
            if agreed_models:
                print(f"Agreed: {', '.join(agreed_models)}")
            if disagreed_models:
                print(f"Critiqued: {', '.join(disagreed_models)}")

        if user_feedback:
            print()
            print(f"=== User Feedback ===")
            print(user_feedback)


if __name__ == "__main__":
    main()
