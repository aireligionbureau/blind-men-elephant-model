# Blind Men and the Elephant Model

[简体中文](README.md) | English

<p align="center">
  <img src="homepage/assets/blind-men-elephant-model-icon.png" alt="An elephant assembled from incomplete puzzle pieces" width="180">
</p>

In the parable, several blind people touch an elephant. One feels a leg and calls it a pillar; another feels the trunk and calls it a snake. Each has encountered something real, but none has encountered the whole animal.

Complex questions are like that elephant. Our limits, blind spots, and biases are not merely errors to discard: they are clues. The Blind Men and the Elephant Model studies the *observers* as well as their answers, using the shadows in their thinking to sketch a provisional outline of the larger problem.

[![A 15-second transition from scattered puzzle pieces to an incomplete elephant and a real result page](docs/media/ai-religion-transition.webp)](docs/media/ai-religion-transition.mp4)

*A run exploring whether AI could develop a religion of its own. [Watch the 15-second video](docs/media/ai-religion-transition.mp4).*

> They touch the elephant. We examine how they touched it.

## What the model does

For each new question, the system generates a new, deliberately diverse cohort of AI observers: 24 in standard mode or 50 in deep mode. Each observer has a structured cognitive identity, including an analytical approach, knowledge domain, values, time horizon, risk attitude, information preferences, and potential biases. The identities are generated for the question; the same cast is not reused across unrelated questions.

Observers begin with their own assumptions, search through their own information filters, decide what to accept or reject, and explain how they reached a conclusion. An evidence ledger records that path. The meta-model then works in three connected stages:

1. **Examine each reasoning chain:** What did this observer look for, trust, ignore, assume, infer, and conclude? Where did the reasoning narrow, jump, or fail?
2. **Connect the observations:** Which apparent disagreements come from different definitions, time horizons, evidence, or values? Where do different observers support, correct, or expose gaps in one another's accounts?
3. **Draw a provisional contour:** State the main answer in ordinary language, the conditions under which it would change, and what the run still cannot establish.

The output is **not** a vote, a debate winner, or a claim to a final truth. It aims to give the reader a more defensible overall outline than any one locally precise viewpoint can provide. The result page also lets readers inspect how individual observers searched and reasoned, and what the meta-model found problematic in their thinking.

## Try it on Windows

1. Download the repository ZIP and extract it fully.
2. Double-click `install-windows.cmd` in the extracted folder.
3. In the browser setup page, choose DeepSeek, OpenAI, or another compatible provider; select a model, enter your own API key, and click the verification/start button.

The installer checks for Python 3.10 or later and can offer to install Python 3.12 through Windows Package Manager if needed. It creates a project-specific virtual environment and a desktop shortcut. On later visits, use the shortcut or `start-windows.cmd`. Moving the extracted folder requires running `install-windows.cmd` again to refresh the shortcut; saved settings and results are preserved.

The web setup does not require editing source files or setting environment variables. On Windows, the key is encrypted for the current account with DPAPI and saved outside the repository. It is not written to reports, browser storage, or service logs; the settings API returns only its last four characters. Changing the provider or endpoint requires a new key rather than silently sending an old key to a new service.

## Run from source

Python 3.10 or later is required. From the repository root:

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install --editable .
python tools/run_plain_tests.py
```

To run the local web app from the command line, provide a model API key and start the server:

```powershell
$env:DEEPSEEK_API_KEY="your-api-key"
python -m bme_model serve --host 127.0.0.1 --port 8765 --runs-dir runs --provider three-layer
```

Open `http://127.0.0.1:8765` in your browser. The Windows installer and launcher are the supported one-click route; on other operating systems, use the Python command-line route.

To use another OpenAI Chat Completions-compatible provider from the command line:

```powershell
$env:BME_LLM_API_KEY="your-api-key"
$env:BME_LLM_PROVIDER_PRESET="openai-compatible"
$env:BME_LLM_BASE_URL="https://your-provider.example/v1"
$env:BME_LLM_MODEL="your-model"
$env:BME_COHORT_MODEL="your-cohort-model"
$env:BME_LLM_SUPPORTS_THINKING="false"
```

The endpoint must support the project's Chat Completions, tool-calling, and JSON-output needs. A provider's presence in the setup page is not a guarantee that every model it offers is compatible. API usage is billed by your provider; duration and cost vary with the question, model, search availability, and provider load. Optional search API credentials can be supplied separately; see the [Chinese README](README.md#模型接口) for the full configuration reference.

The default `plan` command is offline and does not spend model tokens:

```bash
python -m bme_model plan "Should our city expand night bus service?"
```

## How to read a result

The contour is a **working interpretation**, not a verified ground truth. Some retrieved items are only search snippets; they can expose an observer's information diet or suggest a hypothesis, but cannot by themselves establish a real-world fact. Reposted articles are not independent evidence. The observers may share biases because they rely on the same underlying model. Missing evidence must remain visible as an unknown, not be filled in with confidence.

The classic works in the knowledge base are used as diagnostic lenses to test explanations of cognitive shadows. They do not override external evidence or dictate the connections between observers. The meta-model's own preference for tidy, rational explanations is also a possible blind spot.

This is an open-source research system for exploring complex questions, not a substitute for domain experts, primary-source verification, or accountable human decisions. It records its run artifacts and supports recovery after interrupted runs, but it does not promise that every run will finish within a fixed time under arbitrary network or provider conditions.

## Repository map

```text
docs/       Theory, design decisions, and audits (primarily in Chinese)
knowledge/  Diagnostic lenses distilled from classic works
schemas/    Structures for observers, evidence, shadows, relations, and contours
prompts/    Observer and meta-model protocols
src/        Python implementation
examples/   Example questions and report sketches
tests/      Offline tests
```

For the detailed method, architecture, performance records, and complete command reference, see the [Chinese README](README.md) and the documents in [`docs/`](docs/).
