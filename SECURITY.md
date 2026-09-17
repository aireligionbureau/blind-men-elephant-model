# Security

## Reporting a Vulnerability

Please use the repository's GitHub **Security** tab and choose **Report a vulnerability**. Do not open a public issue for an unpatched vulnerability. Include affected versions, reproduction steps, impact, and a minimal proof of concept. Remove API keys, authorization headers, personal data, and unrelated provider responses before submitting.

Maintainers will acknowledge a complete report through the private advisory and coordinate disclosure after a fix or mitigation is available.

## Secrets

Do not commit API keys, `.env` files, raw provider responses containing secrets,
or execution logs that may include authorization headers.

The local web setup stores model credentials outside the repository. On Windows,
the API key is protected with current-user DPAPI and written to:

```text
%APPDATA%\BlindMenElephantModel\settings.json
```

The settings API never returns the complete key, and the browser does not put it
in local storage. Reports and service logs must not contain it. Deleting that
settings file removes the saved web configuration.

Command-line and automated deployments can instead use environment variables:

```powershell
$env:DEEPSEEK_API_KEY="your-api-key"
```

For another OpenAI-compatible provider, use `BME_LLM_API_KEY`,
`BME_LLM_BASE_URL`, and `BME_LLM_MODEL`. Never commit these values.

For search providers:

```powershell
$env:GOOGLE_API_KEY="your-google-key"
$env:GOOGLE_CSE_ID="your-google-cse-id"
$env:BAIDU_SEARCH_API_KEY="your-baidu-or-serp-key"
```

The repository includes `.env.example` only as a template.

## Untrusted Search Content

Search titles, snippets, and fetched text are untrusted data. Instructions found
inside them must never change a digital person's identity, the current question,
the system prompt, tool policy, or output schema. Keep provider responses out of
system messages and preserve their source lineage in the evidence ledger.
