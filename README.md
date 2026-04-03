# Gemma 4 FiftyOne Zoo Model

A FiftyOne remote zoo model integration for [Google Gemma 4](https://ai.google.dev/gemma), a multimodal vision-language model family supporting **image** and **video** understanding.

Structured operations (detect, point, classify) use Gemma 4's native **function calling** for reliable structured output. Text operations (vqa, caption, ocr) use plain generation. All outputs go through `parse_response` for clean separation of thinking and content.

## Quick Start

```python
import fiftyone as fo
import fiftyone.zoo as foz

# Register the model source
foz.register_zoo_model_source(
    "https://github.com/Burhan-Q/gemma4",
    overwrite=True,
)

# Download the model weights
foz.download_zoo_model(
    "https://github.com/Burhan-Q/gemma4",
    model_name="google/gemma-4-E4B-it",
)

# Load a dataset and run inference
dataset = foz.load_zoo_dataset("quickstart")

model = foz.load_zoo_model(
    "google/gemma-4-E4B-it",
    media_type="image",
    operation="vqa",
)

model.prompt = "Describe what is happening in this image."
dataset.apply_model(model, label_field="description")

session = fo.launch_app(dataset)
```

---

## Verifying Your Setup

The included `examples.py` provides minimal, self-contained tests for each task type. Use it to quickly verify that Gemma 4 inference is working correctly on your machine.

```bash
# Run a single task
uv run examples.py detect

# Run all image tasks
uv run examples.py all
```

Available tasks: `vqa`, `caption`, `ocr`, `detect`, `point`, `classify`, `video_description`, `video_custom`, `all`

Each task loads the smallest possible dataset slice (typically 2 samples), runs inference, and prints the full results including geometry data for spatial operations. Video tasks require `ffprobe` (install via `brew install ffmpeg` on macOS).

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

## Installation

```bash
pip install fiftyone transformers>=4.52.0 torch torchvision accelerate huggingface-hub
```

For video processing, you also need `torchcodec` and `ffmpeg`:

```bash
pip install torchcodec

# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

---

## Supported Models

| Model | Effective Params | Context | Modalities | VRAM (est.) |
|-------|-----------------|---------|------------|-------------|
| `google/gemma-4-E2B-it` | 2.3B (5.1B total) | 128K | Text / Image / Video / Audio | ~10 GB |
| `google/gemma-4-E4B-it` | 4.5B (8B total) | 128K | Text / Image / Video / Audio | ~16 GB |
| `google/gemma-4-26B-A4B-it` | 3.8B active (25.2B MoE) | 256K | Text / Image | ~50 GB |
| `google/gemma-4-31B-it` | 30.7B dense | 256K | Text / Image | ~62 GB |

**Notes:**
- Only the **E2B** and **E4B** models support video and audio input. The 26B-A4B and 31B models are image-only.
- The **26B-A4B** model (MoE architecture) requires **CUDA**. It does not currently run on Apple Silicon MPS due to a missing `torch.histc` implementation for the MoE expert routing layer.
- All instruction-tuned models use the `-it` suffix. Base (pre-trained) models exist but are not supported by this integration since they lack the chat template and tool calling capabilities.

---

## Image Operations

Load the model with `media_type="image"` (the default). Switch operations by setting `model.operation`.

### Visual Question Answering

```python
model = foz.load_zoo_model("google/gemma-4-E4B-it", operation="vqa")
model.prompt = "What objects are visible in this image?"

dataset.apply_model(model, label_field="q_vqa")

print(dataset.first().q_vqa)  # fo.Classification with label=text
```

**Output**: `fo.Classification`

### Caption

```python
model = foz.load_zoo_model("google/gemma-4-E4B-it", operation="caption")
model.prompt = "Describe this image in one sentence."

dataset.apply_model(model, label_field="caption")
```

**Output**: `fo.Classification`

### Object Detection

Uses function calling with the `report_detections` tool. The model outputs bounding boxes in its native `box_2d` format with `[y1, x1, y2, x2]` coordinates scaled 0-1000, which are automatically converted to FiftyOne's `[x, y, w, h]` in `[0, 1]` range.

```python
model = foz.load_zoo_model("google/gemma-4-E4B-it", operation="detect")
model.prompt = "Detect all objects."

dataset.apply_model(model, label_field="dets")

# Inspect results
for det in dataset.first().dets.detections:
    print(det.label, det.bounding_box)  # [x, y, w, h] normalized
```

**Output**: `fo.Detections`

### Keypoint Detection

Uses function calling with the `report_points` tool. Coordinates follow the same `[y, x]` native format, auto-converted to `[x, y]` for FiftyOne.

```python
model = foz.load_zoo_model("google/gemma-4-E4B-it", operation="point")
model.prompt = "Point to the center of each animal in this image."

dataset.apply_model(model, label_field="pts")

for kp in dataset.first().pts.keypoints:
    print(kp.label, kp.points)  # [[x, y]] normalized
```

**Output**: `fo.Keypoints`

### Image Classification

Uses function calling with the `report_classifications` tool.

```python
model = foz.load_zoo_model("google/gemma-4-E4B-it", operation="classify")
model.prompt = "Classify this image."

dataset.apply_model(model, label_field="cls")

for c in dataset.first().cls.classifications:
    print(c.label)
```

**Output**: `fo.Classifications`

### OCR

Best results with document images. Use `max_soft_tokens=560` or higher for fine text.

```python
from fiftyone.utils.huggingface import load_from_hub

dataset = load_from_hub("Voxel51/visual_ai_at_neurips2025", max_samples=2)

model = foz.load_zoo_model(
    "google/gemma-4-E4B-it",
    operation="ocr",
    max_soft_tokens=560,  # higher resolution for text extraction
)
model.prompt = "Extract all visible text from this image."

dataset.apply_model(model, label_field="text")

print(dataset.first().text.label)
```

**Output**: `fo.Classification`

### Per-Sample Prompts

Use a dataset field as the prompt source for each sample:

```python
# First, generate descriptions
model.operation = "vqa"
model.prompt = "List all objects in this image."
dataset.apply_model(model, label_field="objects")

# Then use those descriptions to ground detection
model.operation = "detect"
dataset.apply_model(
    model,
    label_field="grounded_dets",
    prompt_field="objects",
)
```

### Overriding the System Prompt

Each operation has a default system prompt. Override it for custom behavior:

```python
model = foz.load_zoo_model("google/gemma-4-E4B-it", operation="detect")
model.system_prompt = "You are a quality inspector. Detect all defects..."
model.prompt = "Find any scratches, dents, or discoloration."

dataset.apply_model(model, label_field="defects")
```

---

## Video Operations

Load the model with `media_type="video"`. Only E2B and E4B models support video. Operations that produce temporal or frame-level labels require `dataset.compute_metadata()`.

**Prerequisite**: Video processing requires `ffprobe` (part of `ffmpeg`).

```python
video_dataset = foz.load_zoo_dataset("quickstart-video")
video_dataset.compute_metadata()

model = foz.load_zoo_model(
    "google/gemma-4-E4B-it",
    media_type="video",
)
```

### Description

Plain-text summary. Does not require metadata.

```python
model.operation = "description"
video_dataset.apply_model(model, label_field="desc")
# result: sample.desc_summary (str)
```

### Temporal Localization

Detects activity events with start/end timestamps.

```python
model.operation = "temporal_localization"
video_dataset.apply_model(model, label_field="events")
# result: sample.events_events (fo.TemporalDetections)
```

### Object Tracking

Tracks objects across frames with per-frame bounding boxes.

```python
model.operation = "tracking"
video_dataset.apply_model(model, label_field="tracking")
# result: sample.frames[N].tracking_objects (fo.Detections)
```

### Video OCR

Extracts text with bounding boxes per frame.

```python
model.operation = "ocr"
video_dataset.apply_model(model, label_field="vocr")
# result: sample.frames[N].vocr_text_content (fo.Detections)
```

### Comprehensive Analysis

All analyses in a single pass: summary, events, objects, scene info, activities.

```python
model.operation = "comprehensive"
video_dataset.apply_model(model, label_field="analysis")
```

### Custom Prompts

Full control over the prompt for domain-specific analysis.

```python
model = foz.load_zoo_model(
    "google/gemma-4-E4B-it",
    media_type="video",
    operation="custom",
    custom_prompt="Count the number of people entering and leaving the frame.",
)

video_dataset.apply_model(model, label_field="count")
# result: sample.count_result (str)
```

---

## Thinking Mode

Gemma 4 supports a reasoning mode where the model shows step-by-step thinking before its answer.

```python
model = foz.load_zoo_model(
    "google/gemma-4-E4B-it",
    operation="detect",
    enable_thinking=True,
)

model.prompt = "Detect all road signs in this image."
dataset.apply_model(model, label_field="signs")

# Reasoning is stored as a dynamic attribute on each label
det = dataset.first().signs.detections[0]
print(det.label)
print(det["reasoning"])  # model's thinking chain, if present
```

**Important**: Thinking mode significantly increases token usage and inference time. For structured operations (detect, point, classify), thinking can cause the model to exhaust its generation budget before producing the tool call. It is recommended to keep `enable_thinking=False` (the default) for structured operations and increase `max_new_tokens` if you do enable it.

---

## Configuration Parameters

All parameters can be set at load time or modified after loading via properties.

```python
model = foz.load_zoo_model(
    "google/gemma-4-E4B-it",
    operation="detect",
    max_new_tokens=4096,
    temperature=0.7,
    max_soft_tokens=560,
)

# Or modify after loading
model.max_new_tokens = 2048
model.temperature = 0.5
```

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 2048 | Maximum tokens to generate. Must be high enough for model thinking (if enabled) plus the response. |
| `temperature` | 1.0 | Sampling temperature |
| `top_p` | 0.95 | Nucleus sampling threshold |
| `top_k` | 64 | Top-k sampling parameter |
| `do_sample` | True | Sampling (True) vs greedy decoding (False) |
| `repetition_penalty` | 1.0 | Penalize repeated tokens |
| `enable_thinking` | False | Enable step-by-step reasoning mode. See Thinking Mode section for caveats. |
| `cache_implementation` | None | KV cache strategy for `generate()`. `"static"` pre-allocates cache (used in official Gemma 4 examples). May not work with all model variants (e.g., 26B MoE). |

### Vision Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_soft_tokens` | varies | Vision token budget per image. Must be one of: **70**, **140**, **280**, **560**, **1120**. Default is operation-dependent (see below). |

Operation-specific defaults for `max_soft_tokens`:

| Operation | Default | Rationale |
|-----------|---------|-----------|
| detect, point | 280 | Balanced detail for object localization |
| classify, vqa, caption | 280 | General-purpose |
| ocr | 560 | Higher resolution needed for text extraction |

Override for your use case:

```python
# Maximum detail for document OCR
model = foz.load_zoo_model(..., operation="ocr", max_soft_tokens=1120)

# Fast detection with lower resolution
model = foz.load_zoo_model(..., operation="detect", max_soft_tokens=140)
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

---

## Citation

```bibtex
@misc{gemma4,
  title  = {Gemma 4 Technical Report},
  author = {Google DeepMind},
  year   = {2026},
  url    = {https://ai.google.dev/gemma}
}
```
