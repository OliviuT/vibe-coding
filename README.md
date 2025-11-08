# vibe-coding

A repository for vibe-coding projects.

## Telemetry + ChatGPT example

This project includes `telemetry_system.py`, a small script that collects a
snapshot of system telemetry and sends it to OpenAI's Chat Completions API for
analysis.

### Requirements

* Python 3.9+
* An OpenAI API key with access to Chat Completions (`OPENAI_API_KEY`).

### Usage

Collect telemetry without calling the API:

```bash
python telemetry_system.py --collect-only
```

Send the telemetry to ChatGPT for analysis:

```bash
export OPENAI_API_KEY="sk-..."
python telemetry_system.py
```

Optional flags:

* `--model`: override the ChatGPT model (defaults to `gpt-4o-mini`).
* `--endpoint`: change the API endpoint if you are using a proxy.
* `--timeout`: configure the request timeout (seconds).

The script prints the model's analysis. To inspect the full JSON response, modify
`telemetry_system.py` to handle `result["raw_response"]` as needed.
