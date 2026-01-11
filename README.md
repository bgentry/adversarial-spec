# adversarial-spec

A Claude Code plugin that iteratively refines product specifications through multi-model debate until consensus is reached.

**Key insight:** A single LLM reviewing a spec will miss things. Multiple LLMs debating a spec will catch gaps, challenge assumptions, and surface edge cases that any one model would overlook. The result is a document that has survived rigorous adversarial review.

**Claude is an active participant**, not just an orchestrator. Claude provides independent critiques, challenges opponent models, and contributes substantive improvements alongside external models.

## Quick Start

```bash
# 1. Install the plugin
claude plugin add github:zscole/adversarial-spec

# 2. Set at least one API key
export OPENAI_API_KEY="sk-..."

# 3. Run it
/adversarial-spec "Build a rate limiter service with Redis backend"
```

## How It Works

```
You describe product ─► Claude drafts spec ─► Multiple LLMs critique in parallel
        │                                              │
        │                                              ▼
        │                              Claude synthesizes + adds own critique
        │                                              │
        │                                              ▼
        │                              Revise and repeat until ALL agree
        │                                              │
        └─────────────────────────────────────────────►│
                                                       ▼
                                            User review period
                                                       │
                                                       ▼
                                            Final document output
```

1. Describe your product concept or provide an existing document
2. (Optional) Start with an in-depth interview to capture requirements
3. Claude drafts the initial document (PRD or tech spec)
4. Document is sent to opponent models (GPT, Gemini, Grok, etc.) for parallel critique
5. Claude provides independent critique alongside opponent feedback
6. Claude synthesizes all feedback and revises
7. Loop continues until ALL models AND Claude agree
8. User review period: request changes or run additional cycles
9. Final converged document is output

## Requirements

- Python 3.10+
- `litellm` package: `pip install litellm`
- API key for at least one LLM provider

## Supported Models

| Provider  | Env Var              | Example Models                               |
|-----------|----------------------|----------------------------------------------|
| OpenAI    | `OPENAI_API_KEY`     | `gpt-4o`, `gpt-4-turbo`, `o1`                |
| Google    | `GEMINI_API_KEY`     | `gemini/gemini-2.0-flash`, `gemini/gemini-pro` |
| xAI       | `XAI_API_KEY`        | `xai/grok-3`, `xai/grok-beta`                |
| Mistral   | `MISTRAL_API_KEY`    | `mistral/mistral-large`, `mistral/codestral` |
| Groq      | `GROQ_API_KEY`       | `groq/llama-3.3-70b-versatile`               |
| Deepseek  | `DEEPSEEK_API_KEY`   | `deepseek/deepseek-chat`                     |

Check which keys are configured:

```bash
python3 ~/.claude/skills/adversarial-spec/scripts/debate.py providers
```

## Usage

**Start from scratch:**

```
/adversarial-spec "Build a rate limiter service with Redis backend"
```

**Refine an existing document:**

```
/adversarial-spec ./docs/my-spec.md
```

You will be prompted for:

1. **Document type**: PRD (business/product focus) or tech spec (engineering focus)
2. **Interview mode**: Optional in-depth requirements gathering session
3. **Opponent models**: Comma-separated list (e.g., `gpt-4o,gemini/gemini-2.0-flash,xai/grok-3`)

More models = more perspectives = stricter convergence.

## Document Types

### PRD (Product Requirements Document)

For stakeholders, PMs, and designers.

**Sections:** Executive Summary, Problem Statement, Target Users/Personas, User Stories, Functional Requirements, Non-Functional Requirements, Success Metrics, Scope (In/Out), Dependencies, Risks

**Critique focuses on:** Clear problem definition, well-defined personas, measurable success criteria, explicit scope boundaries, no technical implementation details

### Technical Specification

For developers and architects.

**Sections:** Overview, Goals/Non-Goals, System Architecture, Component Design, API Design (full schemas), Data Models, Infrastructure, Security, Error Handling, Performance/SLAs, Observability, Testing Strategy, Deployment Strategy

**Critique focuses on:** Complete API contracts, data model coverage, security threat mitigation, error handling, specific performance targets, no ambiguity for engineers

## Features

### Interview Mode

Before the debate begins, opt into an in-depth interview session to capture requirements upfront.

**Covers:** Problem context, users/stakeholders, functional requirements, technical constraints, UI/UX, tradeoffs, risks, success criteria

The interview uses probing follow-up questions and challenges assumptions. After completion, Claude synthesizes answers into a complete spec before starting the adversarial debate.

### Claude's Active Participation

Each round, Claude:

1. Reviews opponent critiques for validity
2. Provides independent critique (what did opponents miss?)
3. States agreement/disagreement with specific points
4. Synthesizes all feedback into revisions

Display format:

```
--- Round N ---
Opponent Models:
- [GPT-4o]: critiqued: missing rate limit config
- [Gemini]: agreed

Claude's Critique:
Security section lacks input validation strategy. Adding OWASP top 10 coverage.

Synthesis:
- Accepted from GPT-4o: rate limit configuration
- Added by Claude: input validation, OWASP coverage
- Rejected: none
```

### Early Agreement Verification

If a model agrees within the first 2 rounds, Claude is skeptical. The model is pressed to:

- Confirm it read the entire document
- List specific sections reviewed
- Explain why it agrees
- Identify any remaining concerns

This prevents false convergence from models that rubber-stamp without thorough review.

### User Review Period

After all models agree, you enter a review period with three options:

1. **Accept as-is**: Document is complete
2. **Request changes**: Claude updates the spec, you iterate without a full debate cycle
3. **Run another cycle**: Send the updated spec through another adversarial debate

### Additional Review Cycles

Run multiple cycles with different strategies:

- First cycle with fast models (gpt-4o), second with stronger models (o1)
- First cycle for structure/completeness, second for security focus
- Fresh perspective after user-requested changes

### PRD to Tech Spec Flow

When a PRD reaches consensus, you're offered the option to continue directly into a Technical Specification based on the PRD. This creates a complete documentation pair in a single session.

## Telegram Integration (Optional)

Get notified on your phone and inject feedback during the debate.

**Setup:**

1. Message @BotFather on Telegram, send `/newbot`, follow prompts
2. Copy the bot token
3. Run: `python3 ~/.claude/skills/adversarial-spec/scripts/telegram_bot.py setup`
4. Message your bot, run setup again to get your chat ID
5. Set environment variables:

```bash
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
```

**Features:**

- Async notifications when rounds complete
- 60-second window to reply with feedback (incorporated into next round)
- Final document sent to Telegram when debate concludes

## Output

Final document is:

- Complete, following full structure for document type
- Vetted by all models until unanimous agreement
- Ready for stakeholders without further editing

Output locations:

- Printed to terminal
- Written to `spec-output.md` (PRD) or `tech-spec-output.md` (tech spec)
- Sent to Telegram (if enabled)

Debate summary includes rounds completed, cycles run, models involved, Claude's contributions, and key refinements made.

## File Structure

```
adversarial-spec/
├── .claude-plugin/
│   └── plugin.json           # Plugin metadata
├── README.md
├── LICENSE
└── skills/
    └── adversarial-spec/
        ├── SKILL.md          # Skill definition and process
        └── scripts/
            ├── debate.py     # Multi-model debate orchestration
            └── telegram_bot.py   # Telegram notifications
```

## License

MIT
