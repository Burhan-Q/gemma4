# Additional Information

Supplementary documentation for the [Gemma 4 FiftyOne Zoo Model](README.md).

---

## Table of Contents

- [Verifying Your Setup](#verifying-your-setup)
- [Architecture](#architecture)
- [Logging](#logging)
- [Technical Details](#technical-details)

---

## Verifying Your Setup

The included `examples.py` provides minimal, self-contained tests for each task type. Use it to quickly verify that Gemma 4 inference is working correctly on your machine.

```bash
# Using uv
uv run examples.py detect

# Using standard Python
python examples.py detect

# Run all image tasks
uv run examples.py all
# or
python examples.py all
```

Available tasks: `vqa`, `caption`, `ocr`, `detect`, `point`, `classify`, `video_description`, `video_custom`, `all`

Each task loads the smallest possible dataset slice (typically 2 samples), runs inference, and prints the full results including geometry data for spatial operations. Video tasks require `ffprobe` (see [Installation](README.md#installation) for setup).

Example output for detection:

```
=== detect ===
  000880.jpg: 2 detections
    label=wild turkey  bbox=[0.214, 0.013, 0.438, 0.768]
    label=wild turkey  bbox=[0.751, 0.488, 0.22, 0.214]

  001599.jpg: 2 detections
    label=person  bbox=[0.602, 0.034, 0.233, 0.597]
    label=horse  bbox=[0.178, 0.284, 0.822, 0.715]

  [OK] Detect
```

---

## Architecture

### Inference Pipeline

```
apply_chat_template (with tools= for structured ops)
  -> model.generate
  -> processor.decode (skip_special_tokens=False)
  -> processor.parse_response -> {role, thinking, content, tool_calls}
  -> extract tool_calls arguments (structured) or content text (free-form)
  -> convert to FiftyOne Label
```

All inference goes through `parse_response`, which separates thinking traces from content and tool calls. For structured operations, tool calling produces pre-parsed dicts that don't require JSON extraction from free text.

### Gemma 4 Native Formats

The model uses specific native formats for spatial outputs that differ from some other VLMs:

- **Bounding boxes**: `box_2d` key with `[y1, x1, y2, x2]` coordinates in 0-1000 scale
- **Points**: `point_2d` key with `[y, x]` coordinates in 0-1000 scale

These are automatically converted to FiftyOne's standard formats:
- Detections: `[x, y, width, height]` in `[0, 1]` range
- Keypoints: `[[x, y]]` in `[0, 1]` range

### Structured Operations (detect, point, classify)

These operations pass tool definitions to `apply_chat_template`, leveraging Gemma 4's native function calling. The model produces structured tool call arguments that are already parsed dicts. If `parse_response` fails to extract the tool call (e.g., due to malformed JSON in the arguments), a fallback parser attempts to recover the data from the raw output.

### Text Operations (vqa, caption, ocr)

These operations use plain generation without tools. The content text from `parse_response` is wrapped in `fo.Classification(label=text)`.

### Return Types

All image operations return a **single FiftyOne Label instance** (not a dict), compatible with any `label_field` path:

| Operation | FiftyOne Type |
|-----------|--------------|
| `vqa` | `fo.Classification` |
| `caption` | `fo.Classification` |
| `ocr` | `fo.Classification` |
| `detect` | `fo.Detections` |
| `point` | `fo.Keypoints` |
| `classify` | `fo.Classifications` |

Video operations return dicts (sample-level and/or frame-level labels).

---

## Logging

Logging is controlled via environment variables. By default, the logger outputs at `INFO` level to stderr.

| Variable | Default | Description |
|----------|---------|-------------|
| `FIFTYONE_GEMMA4_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `FIFTYONE_GEMMA4_LOGFILE` | (unset) | Set to `1`, `true`, or `True` to enable logging to file |

Log files are named `run-001.log`, `run-002.log`, etc. in the current working directory, auto-incrementing.

```bash
# Verbose console logging
FIFTYONE_GEMMA4_LOG_LEVEL=DEBUG python my_script.py

# Enable file logging (creates run-001.log, etc.)
FIFTYONE_GEMMA4_LOGFILE=1 python my_script.py

# Both: debug to console + file
FIFTYONE_GEMMA4_LOG_LEVEL=DEBUG FIFTYONE_GEMMA4_LOGFILE=1 python my_script.py
```

At `DEBUG` level, the full raw model output is logged for each sample -- useful for diagnosing parsing failures.

---

## Technical Details

- **Model class**: Uses `AutoModelForImageTextToText` from `transformers` (not `AutoModelForMultimodalLM`), matching the official transformers documentation for Gemma 4.
- **Coordinate system**: Gemma 4 outputs coordinates as `[y1, x1, y2, x2]` in 0-1000 scale using the `box_2d` key. These are auto-converted to FiftyOne's `[x, y, w, h]` in `[0, 1]` range.
- **dtype**: `bfloat16` on Ampere+ GPUs (CUDA compute capability >= 8.0), `auto` otherwise. MPS and CPU are also supported for dense models.
- **MoE limitation**: The 26B-A4B model (Mixture-of-Experts) requires CUDA. It cannot run on MPS due to an unimplemented `torch.histc` operation in the expert routing layer.
- **Video handling**: Gemma 4 processes video natively -- no frame extraction needed. Video is limited to 60 seconds per the model specification. Requires `ffprobe`.
- **Metadata requirement**: Video operations that produce temporal or frame-level labels require `dataset.compute_metadata()`.
- **Inference**: Samples are processed sequentially (one at a time) to manage GPU memory.
- **DataLoader compatibility**: The collate function is defined at module level for pickle compatibility with multiprocessing DataLoader workers.
- **JSON repair**: Minimal repair is applied for trivial JSON errors (double commas, trailing commas, double-wrapped brackets). Fundamentally malformed JSON from the model is not repaired -- that is a model quality issue.
- **Free-form JSON is unreliable**: Without tool calling, the model frequently produces garbled JSON (wrong keys, unquoted values, broken structure). Tool calling is essential for reliable structured output from Gemma 4.
