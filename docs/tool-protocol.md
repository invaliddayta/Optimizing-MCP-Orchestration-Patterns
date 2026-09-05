# Tool Protocol Evidence

[Back to the README](../README.md)

An OpenAI-compatible JSON API does not establish native tool-use training.
llama-server renders messages using a model-specific template and parses the
model's output. It can also fall back to a generic format. `--jinja` by itself
does not tell us which handler was selected.

## Observations And Declarations

The harness records templates and their SHA-256 hashes, `build_info`, and
`model_path` from the server's read-only `/props` endpoint. For an API base like
`http://host/prefix/v1`, it uses `http://host/prefix/props`. Unavailable properties
are recorded as unavailable, not guessed. No inference about training is made
from the model name, template text, or successful probes.

The pinned llama.cpp build reports the selected handler in its logs (for example,
`Chat format: Generic`); `/props` does not expose that choice. Inspect the log for
the requests being tested, and use a model with documented tool support and the
corresponding tool-enabled template. The pinned build's
[function-calling guide](https://github.com/ggml-org/llama.cpp/blob/b6981/docs/function-calling.md)
lists Qwen2.5 Instruct with its tool template as a native-supported starting point.
Handler support alone is not evidence about a particular fine-tune's training.

## Metadata Format

By default, smoke runs without evidence are allowed but emit warnings and are
labelled `unverified`. For evidence-backed experiments, provide
`--model-metadata metadata.json` with a `tool_protocols` array. Every active
endpoint/model pair needs its own declaration:

```json
{
  "tool_protocols": [
    {
      "base_url": "http://127.0.0.1:8080/v1",
      "model": "local",
      "format": "native",
      "handler": "Hermes 2 Pro",
      "handler_source": "Reference to the captured server log for this experiment",
      "training_source": "Versioned model card or training documentation for this exact model",
      "gguf_sha256": "REPLACE_WITH_64_LOWERCASE_HEX_CHARACTERS",
      "quantization": "Q4_K_M",
      "build_info": "REPLACE_WITH_THE_OBSERVED_BUILD_INFO",
      "chat_template_sha256": "REPLACE_WITH_THE_OBSERVED_TEMPLATE_SHA256",
      "chat_template_tool_use_sha256": null
    }
  ]
}
```

This is a template, not a ready-to-use model profile. Run `doctor` without
`--require-native` first to collect observed hashes and build information. Hash
your actual GGUF separately. Declare the tool-use template's hash too when
`/props` exposes one; use `null` only when it is absent. Keep launch flags,
hardware, and other notes in additional metadata fields.

For a generic-format experiment, set `format` to `generic` and `handler` to
`Generic`; a training reference is not required for that condition.

## Strict Runs

```sh
python -m lab doctor --model local --model-metadata metadata.json --require-native
python -m lab run --model local --limit 3 --model-metadata metadata.json \
  --require-native --output results/native-smoke
python -m lab trace results/native-smoke --preflight
```

`--require-native` rejects generic declarations, absent evidence, and template or
build mismatches before inference. It checks manager and worker independently;
passing one model's probes does not excuse another. Matching endpoint/model pairs
are probed only once. If strict `doctor` rejects a declaration, it prints the
evidence to stderr.

The behavioral probes separately test forced calling, automatic tool selection,
consumption of an external result, and answering without unnecessary tool calls.
Failed probes preserve the stage, request and response in the event log;
`trace --preflight` displays these alongside provenance and setup errors.

## Interpretation

Reports carry a `protocol_condition` per mode, with `native_declared`,
`generic_declared`, or `unverified` labels for each participating model role.
Runs from the earlier harness revision without provenance remain `unverified`,
even if they contain the former `native_tool_roundtrip` flag. Do not pool different
conditions as if they were the same experiment.

**Verification is deliberately limited:** template hashes and the server's build
string are compared; GGUF identity, handler selection, and training references
remain user-declared. A server file path is not a weight hash. `native_declared`
is not a certificate of training alignment, and `--require-native` does not
independently inspect weights, fetch model cards, or authenticate server logs.
