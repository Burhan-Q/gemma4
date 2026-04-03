# Gemma 4 FiftyOne Zoo Model

<div align="center">
  <a href="https://github.com/voxel51/fiftyone"><img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
  <img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px"></a><br>
  <img src=https://ai.google.dev/gemma/images/gemma4_banner.png width="380px">
</div>

A [FiftyOne](https://github.com/voxel51/fiftyone) remote Model Zoo integration for [Google Gemma 4](https://ai.google.dev/gemma), a multimodal vision-language model family supporting **image** and **video** understanding. Learn more about the [FiftyOne Model Zoo in the Voxel51 docs](https://docs.voxel51.com/model_zoo).

Structured operations (detect, point, classify) use Gemma s4's native **function calling** for reliable structured output. Text operations (vqa, caption, ocr) use plain generation. All outputs go through `parse_response` for clean separation of thinking and content.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Image Operations](#image-operations)
- [Video Operations](#video-operations)
- [Configuration Parameters](#configuration-parameters)
- [Thinking Mode](#thinking-mode)
- [Additional Info](ADDITIONAL_INFO.md) (setup verification, architecture, logging, technical details)
- [Citation](#citation)

---

## Installation

Using `pip`:

```bash
pip install fiftyone "transformers>=4.52.0" torch torchvision accelerate huggingface-hub
```

Or using [`uv`](https://docs.astral.sh/uv/):

```bash
uv add fiftyone "transformers>=4.52.0" torch torchvision accelerate huggingface-hub
```

For video processing, you also need `torchcodec` and `ffmpeg`:

```bash
# pip
pip install torchcodec

# or uv
uv add torchcodec
```

`ffmpeg` must be installed separately as a system package:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg
```

---

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

### ⚠️ Important

Use short prompts for better results, especially for smaller models.

---

## Image Operations

Load the model with `media_type="image"` (the default). Switch operations by setting `model.operation`.

```python
dataset = foz.load_zoo_dataset("quickstart")
dataset.compute_metadata()

model = foz.load_zoo_model(
    "google/gemma-4-E4B-it",
    media_type="image",
)
```

<details><summary>👈 Expand for all image tasks</summary>

### Visual Question Answering (VQA)

<details><summary>Prompt model with questions to answer about images.</summary><p>

```python
model = foz.load_zoo_model("google/gemma-4-E4B-it", operation="vqa")
model.prompt = "What objects are visible in this image?"

dataset.apply_model(model, label_field="q_vqa")

print(dataset.first().q_vqa)  # fo.Classification with label=text
```

**Output**: `fo.Classification`

</details>

---

### Image Captioning

<details><summary>Model provides detailed captioning of image scene</summary><p>

```python
model = foz.load_zoo_model("google/gemma-4-E4B-it", operation="caption")
model.prompt = "Describe this image in one sentence."

dataset.apply_model(model, label_field="caption")
```

**Output**: `fo.Classification`

</details>

---

### Image Object Detection

<details><summary>Model classifies and locates objects spatially in images.</summary><p>

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

</details>

---

### Image Keypoint Detection

<details><summary>Model classifies objects and places single keypoint at object's center</summary><p>

Uses function calling with the `report_points` tool. Coordinates follow the same `[y, x]` native format, auto-converted to `[x, y]` for FiftyOne.

```python
model = foz.load_zoo_model("google/gemma-4-E4B-it", operation="point")
model.prompt = "Point to the center of each animal in this image."

dataset.apply_model(model, label_field="pts")

for kp in dataset.first().pts.keypoints:
    print(kp.label, kp.points)  # [[x, y]] normalized
```

**Output**: `fo.Keypoints`

</details>

---

### Image Classification

<details><summary>Model generates labels for objects in image</summary><p>

Uses function calling with the `report_classifications` tool.

```python
model = foz.load_zoo_model("google/gemma-4-E4B-it", operation="classify")
model.prompt = "Classify this image."

dataset.apply_model(model, label_field="cls")

for c in dataset.first().cls.classifications:
    print(c.label)
```

**Output**: `fo.Classifications`

</details>

---

### Image Optical Character Recognition (OCR)

<details><summary>Model extracts text from images.</summary><p>

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

</details>

---

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

---

### Overriding the System Prompt

Each operation has a default system prompt. Override it for custom behavior:

```python
model = foz.load_zoo_model("google/gemma-4-E4B-it", operation="detect")
model.system_prompt = "You are a quality inspector. Detect all defects..."
model.prompt = "Find any scratches, dents, or discoloration."

dataset.apply_model(model, label_field="defects")
```

</details>

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

---

<details><summary>👈 Expand for all video tasks</summary>

### Video Description

<details><summary>Plain-text summary. Does not require metadata.</summary><p>

```python
model.operation = "description"
video_dataset.apply_model(model, label_field="desc")
# result: sample.desc_summary (str)
```

</details>

---

### Video Temporal Localization

<details><summary>Detects activity events with start/end timestamps.</summary><p>

```python
model.operation = "temporal_localization"
video_dataset.apply_model(model, label_field="events")
# result: sample.events_events (fo.TemporalDetections)
```

</details>

---

### Video Object Tracking

<details><summary>Tracks objects across frames with per-frame bounding boxes.</summary><p>

```python
model.operation = "tracking"
video_dataset.apply_model(model, label_field="tracking")
# result: sample.frames[N].tracking_objects (fo.Detections)
```

</details>

---

### Video OCR

<details><summary>Extracts text with bounding boxes per frame.</summary><p>

```python
model.operation = "ocr"
video_dataset.apply_model(model, label_field="vocr")
# result: sample.frames[N].vocr_text_content (fo.Detections)
```

</details>

---

### Video Comprehensive Analysis

<details><summary>All analyses in a single pass: summary, events, objects, scene info, activities.</summary><p>

```python
model.operation = "comprehensive"
video_dataset.apply_model(model, label_field="analysis")
```

</details>

---

### Video Custom Prompts

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

</details>

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
| `enable_thinking` | False | Enable step-by-step reasoning mode. See [Thinking Mode](#thinking-mode) for caveats. |
| `cache_implementation` | None | KV cache strategy for `generate()`. `"static"` pre-allocates cache (used in official Gemma 4 examples). May not work with all model variants (e.g., 26B MoE). |

### Vision Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_soft_tokens` | varies | Vision token budget per image. Must be one of: **70**, **140**, **280**, **560**, **1120**. Default is operation-dependent (see below). |

Operation-specific defaults for `max_soft_tokens`:

| Operation | Default | Rationale |
|-----------|---------|-----------|
| detect, point, classify | 280 | Balanced detail for object localization |
| vqa, caption | 280 | General-purpose |
| ocr | 560 | Higher resolution needed for text extraction |

Override for your use case:

```python
# Maximum detail for document OCR
model = foz.load_zoo_model(..., operation="ocr", max_soft_tokens=1120)

# Fast detection with lower resolution
model = foz.load_zoo_model(..., operation="detect", max_soft_tokens=140)
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

## Additional Information

See **[ADDITIONAL_INFO.md](ADDITIONAL_INFO.md)** for setup verification, architecture details, logging configuration, and technical details.

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
