# AI Model Options

CubiAI orchestrates multiple external AI services. Profiles declare which providers to use, and the pipeline aborts with a hard error if a required credential or command is missing.

## Segmentation

| Backend                | Strengths                                                  | Requirements |
|------------------------|------------------------------------------------------------|--------------|
| `huggingface-sam`      | High fidelity masks from Segment Anything variants hosted on Hugging Face Inference. | `HF_API_TOKEN` with access to the chosen model, e.g. `facebook/sam-vit-base`. |
| `slic`                 | Lightweight fallback segmentation using `scikit-image`.    | CPU only. |

### Configuring Hugging Face SAM
- Set `HF_API_TOKEN` in your environment before running the CLI.
- Optional profile fields: `segmentation.huggingface.endpoint`, `max_layers`, and `score_threshold`.
- API calls stream the PNG directly to the inference endpoint; the returned base64 masks are converted into PSD layers.
- Fail-fast behaviour: missing tokens or non-2xx responses raise `PipelineStageError` so placeholder layers are never produced.

## Rigging

### LLM Rig Planner
- The default profile uses `rigging.strategy = llm` and targets OpenAI-compatible chat completions.
- Required environment variable: `OPENAI_API_KEY` (or whatever you set via `rigging.llm_api_key_env`).
- `rigging.llm_model` and `rigging.llm_base_url` can point to any provider that honours the Chat Completions protocol (e.g., OpenAI, Azure, Groq, or Hugging Face text-generation-inference with the compatible middleware).
- The LLM receives layer metadata and must respond with JSON defining `parts`, `parameters`, `deformers`, `physics`, and `motions`. Invalid JSON raises `PipelineStageError`.

### Live2D Builder Command
- After the LLM produces rig metadata, CubiAI expects an external builder to generate a real `model.moc3` file.
- Configure `rigging.builder.command` as a list of arguments. The values may include placeholders:
  - `{PSD_PATH}` – absolute path to `layers.psd`.
  - `{RIG_JSON}` – path to the generated rig description JSON.
  - `{WORKSPACE}` – root of the current workspace run.
  - `{OUTPUT_DIR}` – path to `Live2D/` where the builder should write resulting files.
  - Environment variables such as `$CUBISM_CLI` are expanded before substitution; unresolved tokens trigger a failure.
- Example (hypothetical):
  ```yaml
  rigging:
    builder:
      command:
        - "/Applications/CubismEditor.app/Contents/MacOS/cubism-cli"
        - "--psd"
        - "{PSD_PATH}"
        - "--rig-config"
        - "{RIG_JSON}"
        - "--output"
        - "{OUTPUT_DIR}"
  ```
- If the command is empty or the builder fails to emit `model.moc3`, the pipeline raises a hard error. This prevents empty moc files from slipping into exports.

## Additional Processing
- `scipy`/`scikit-image` remain in the stack for matting and fallback segmentation.
- `httpx` provides resilient HTTP communication with backends (timeouts, status handling).

## Operational Notes
- Remote providers introduce latency—plan for 10–30 seconds per request, more for high-resolution inputs.
- Respect rate limits. Hugging Face free-tier throttles concurrent requests; commercial usage may require dedicated endpoints.
- Capture diagnostic logs: the workspace stores `rig_config.json`, builder stdout/stderr, and API metadata for auditability.

Refer to `docs/pipeline.md` for how these services are sequenced, and `README.md` for environment setup commands.
