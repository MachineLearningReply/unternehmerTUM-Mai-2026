# Agentic / LLM Approach

The workshop pack includes a minimal “agentic” pipeline that can use an LLM for:

- generating analyst-friendly explanations
- recommending an operational action (`review`, `step_up_auth`, `block`)
- optionally classifying suspicious messages (teams can extend)

Files:
- `agents/llm.py`: pluggable LLM client (`MockLLMClient` runs offline)
- `agents/tools.py`: retrieval + summarization tools (recent messages, signals)
- `agents/alert_agent.py`: the `AlertAgent` that calls the LLM and returns structured JSON
- `agents/rule_agent.py`: a `RuleProposalAgent` that suggests new rules/features from error cases

## Run modes

By default everything runs offline via the mock client:

```bash
export LLM_MODE=mock
```

To plug in a real LLM via an OpenAI-compatible endpoint:

```bash
export LLM_MODE=openai_compatible
export LLM_BASE_URL="https://YOUR_HOST"
export LLM_API_KEY="YOUR_KEY"
export LLM_MODEL="YOUR_MODEL"
```

If your provider’s API is different, adapt `OpenAICompatibleClient` or write your own adapter implementing `LLMClient`.
