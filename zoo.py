"""
FiftyOne integration for Google Gemma 4 vision-language models.

Supports image and video understanding with the Gemma 4 model family.

Image operations: detect, point, classify, vqa, caption, ocr
Video operations: description, temporal_localization, tracking, ocr,
                  comprehensive, custom

Architecture:
    Gemma4BaseConfig / Gemma4BaseModel
        ├── Gemma4ImageModelConfig / Gemma4ImageModel  (media_type="image")
        └── Gemma4VideoModelConfig / Gemma4VideoModel  (media_type="video")

Inference pipeline:
    apply_chat_template (with tools= for structured ops)
    → model.generate
    → processor.decode (skip_special_tokens=False)
    → processor.parse_response → {role, thinking, content, tool_calls}
    → extract tool_calls arguments (structured) or content text (free-form)
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import fiftyone.utils.torch as fout
from fiftyone.core.models import SupportsGetItem, TorchModelMixin
from fiftyone.utils.torch import GetItem

from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)


# =============================================================================
# Image operation system prompts
# =============================================================================

_TOOL_RULES = (
    "\n\nYou MUST call the tool immediately. "
    "Do not output any text before calling the tool. "
    "Do not describe, reason, or explain — just call the tool.\n"
    "JSON: double quotes, no trailing commas, integers unquoted."
)

DEFAULT_DETECT_SYSTEM_PROMPT = (
    "You are an object detection assistant. "
    "Detect EVERY distinct object in the image. "
    "Each object MUST be a separate entry in the detections array — "
    "if you see 5 objects, return 5 entries. "
    "Call the report_detections tool with ALL detections. "
    "Bounding box format: box_2d as [y1, x1, y2, x2] integers 0-1000."
    + _TOOL_RULES
)

DEFAULT_POINT_SYSTEM_PROMPT = (
    "You are a keypoint detection assistant. "
    "Point to the center of EVERY requested object. "
    "Each object MUST be a separate entry in the points array. "
    "Call the report_points tool with ALL points. "
    "Point format: point_2d as [y, x] integers 0-1000."
    + _TOOL_RULES
)

DEFAULT_CLASSIFY_SYSTEM_PROMPT = (
    "You are an image classification assistant. "
    "Return ALL applicable labels. "
    "Call the report_classifications tool."
    + _TOOL_RULES
)

DEFAULT_VQA_SYSTEM_PROMPT = (
    "You are a helpful assistant. Provide clear and concise answers to "
    "questions about images in natural language English."
)

DEFAULT_CAPTION_SYSTEM_PROMPT = (
    "You are a helpful assistant. Provide a concise, descriptive caption "
    "for the image in natural language English."
)

DEFAULT_OCR_SYSTEM_PROMPT = (
    "You are a helpful assistant specializing in optical character "
    "recognition. Extract all visible text from the image. Return only "
    "the extracted text, preserving the original layout and formatting "
    "as much as possible."
)

IMAGE_OPERATIONS: Dict[str, str] = {
    "detect": DEFAULT_DETECT_SYSTEM_PROMPT,
    "point": DEFAULT_POINT_SYSTEM_PROMPT,
    "classify": DEFAULT_CLASSIFY_SYSTEM_PROMPT,
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT,
    "caption": DEFAULT_CAPTION_SYSTEM_PROMPT,
    "ocr": DEFAULT_OCR_SYSTEM_PROMPT,
}


# =============================================================================
# Tool definitions for structured operations (detect, point, classify)
# =============================================================================

_DETECT_TOOL = {
    "type": "function",
    "function": {
        "name": "report_detections",
        "description": "Report detected objects with bounding boxes.",
        "parameters": {
            "type": "object",
            "properties": {
                "detections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "box_2d": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "[y1, x1, y2, x2] 0-1000",
                            },
                        },
                        "required": ["label", "box_2d"],
                    },
                }
            },
            "required": ["detections"],
        },
    },
}

_POINT_TOOL = {
    "type": "function",
    "function": {
        "name": "report_points",
        "description": "Report detected keypoints.",
        "parameters": {
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "point_2d": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "[x, y] center point 0-1000",
                            },
                        },
                        "required": ["label", "point_2d"],
                    },
                }
            },
            "required": ["points"],
        },
    },
}

_CLASSIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "report_classifications",
        "description": "Report image classification labels.",
        "parameters": {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                        },
                        "required": ["label"],
                    },
                }
            },
            "required": ["labels"],
        },
    },
}

_OPERATION_TOOLS: Dict[str, Optional[list]] = {
    "detect": [_DETECT_TOOL],
    "point": [_POINT_TOOL],
    "classify": [_CLASSIFY_TOOL],
    "vqa": None,
    "caption": None,
    "ocr": None,
}


# =============================================================================
# Video operation prompts
# =============================================================================

VIDEO_OPERATIONS: Dict[str, Dict] = {
    "comprehensive": {
        "prompt": (
            "Analyze this video comprehensively in JSON format:\n\n"
            "{\n"
            '  "summary": "Brief description of the video",\n'
            '  "objects": [{"name": "object name", "first_appears": "mm:ss.ff", '
            '"last_appears": "mm:ss.ff"}],\n'
            '  "events": [{"start": "mm:ss.ff", "end": "mm:ss.ff", '
            '"description": "event description"}],\n'
            '  "text_content": [{"start": "mm:ss.ff", "end": "mm:ss.ff", '
            '"text": "text content"}],\n'
            '  "scene_info": {"setting": "<one-word>", "time_of_day": "<one-word>", '
            '"location_type": "<one-word>"},\n'
            '  "activities": {"primary_activity": "activity name", '
            '"secondary_activities": "comma-separated activities"}\n'
            "}"
        )
    },
    "description": {
        "prompt": "Provide a detailed description of what happens in this video."
    },
    "temporal_localization": {
        "prompt": (
            "Localize activity events in the video. Output start and end timestamp "
            "for each event.\nProvide in JSON format with 'mm:ss.ff' format:\n"
            '[{"start": "mm:ss.ff", "end": "mm:ss.ff", "description": "..."}]'
        )
    },
    "tracking": {
        "prompt": (
            "Track all objects in this video. For each frame where objects appear, "
            "provide:\n"
            "- time: timestamp (mm:ss.ff)\n"
            "- bbox_2d: bounding box as [x_min, y_min, x_max, y_max] in 0-1000 scale\n"
            "- label: object label\n"
            'Output in JSON: [{"time": "mm:ss.ff", "bbox_2d": [...], "label": "..."}, ...]'
        )
    },
    "ocr": {
        "prompt": (
            "Extract all text appearing in this video. For each text instance, provide:\n"
            "- time: timestamp (mm:ss.ff)\n"
            "- text: the actual text content\n"
            "- bbox_2d: bounding box as [x_min, y_min, x_max, y_max] in 0-1000 scale\n"
            'Output in JSON: [{"time": "mm:ss.ff", "text": "...", "bbox_2d": [...]}, ...]'
        )
    },
    "custom": {"prompt": None},
}

_VIDEO_CAPABLE_MODELS = {
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
}


# =============================================================================
# Helpers
# =============================================================================

def get_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _identity_collate(batch):
    """Module-level identity collate (picklable for DataLoader workers)."""
    return batch


# =============================================================================
# Shared GetItem
# =============================================================================

class Gemma4GetItem(GetItem):
    """Extracts filepath, optional per-sample prompt, and metadata."""

    @property
    def required_keys(self) -> List[str]:
        return ["filepath", "metadata"]

    def __call__(self, sample_dict: dict) -> dict:
        return {
            "filepath": sample_dict["filepath"],
            "prompt": sample_dict.get("prompt_field"),
            "metadata": sample_dict.get("metadata"),
        }


# =============================================================================
# Base config
# =============================================================================

class Gemma4BaseConfig(fout.TorchImageModelConfig):
    """Shared configuration: model path and text generation parameters."""

    def __init__(self, d: dict):
        if "raw_inputs" not in d:
            d["raw_inputs"] = True
        super().__init__(d)

        self.model_path = self.parse_string(
            d, "model_path", default="google/gemma-4-E4B-it"
        )
        # Higher default to accommodate model thinking before tool calls
        self.max_new_tokens = self.parse_number(d, "max_new_tokens", default=2048)
        self.do_sample = self.parse_bool(d, "do_sample", default=True)
        self.temperature = self.parse_number(d, "temperature", default=1.0)
        self.top_p = self.parse_number(d, "top_p", default=0.95)
        self.top_k = self.parse_number(d, "top_k", default=64)
        self.repetition_penalty = self.parse_number(
            d, "repetition_penalty", default=1.0
        )
        self.enable_thinking = self.parse_bool(d, "enable_thinking", default=False)
        # Vision token budget per image: 70, 140, 280, 560, 1120
        self.max_soft_tokens = self.parse_number(d, "max_soft_tokens", default=280)
        # KV cache strategy passed to model.generate(). "static" pre-allocates
        # and is used in official Gemma4 examples. None uses the default.
        self.cache_implementation = self.parse_string(
            d, "cache_implementation", default=None
        )


# =============================================================================
# Base model
# =============================================================================

class Gemma4BaseModel(
    fom.Model, fom.SamplesMixin, SupportsGetItem, TorchModelMixin
):
    """Shared base for image and video Gemma 4 zoo models."""

    def __init__(self, config: Gemma4BaseConfig):
        fom.SamplesMixin.__init__(self)
        SupportsGetItem.__init__(self)

        self._preprocess = False
        self.config = config
        self.device = get_device()
        self._fields: dict = {}
        self._model = None
        self._processor = None

    # -- FiftyOne boilerplate --------------------------------------------------

    @property
    def transforms(self):
        return None

    @property
    def preprocess(self) -> bool:
        return self._preprocess

    @preprocess.setter
    def preprocess(self, value: bool):
        self._preprocess = value

    @property
    def ragged_batches(self) -> bool:
        return False

    @property
    def needs_fields(self) -> dict:
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields: dict):
        self._fields = fields

    @property
    def has_collate_fn(self) -> bool:
        return True

    @property
    def collate_fn(self):
        return _identity_collate

    def build_get_item(self, field_mapping=None) -> Gemma4GetItem:
        return Gemma4GetItem(field_mapping=field_mapping)

    # -- Model loading ---------------------------------------------------------

    def _load_model(self):
        """Lazy-load model and processor. bfloat16 on Ampere+ GPUs."""
        logger.info(f"Loading Gemma 4 from {self.config.model_path}")

        model_kwargs: dict = {"device_map": self.device}
        if self.device == "cuda" and torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(self.device)
            model_kwargs["torch_dtype"] = (
                torch.bfloat16 if cap[0] >= 8 else "auto"
            )
        else:
            model_kwargs["torch_dtype"] = "auto"

        self._model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_path, **model_kwargs
        ).eval()

        self._processor = AutoProcessor.from_pretrained(self.config.model_path)

        if hasattr(self._processor, "tokenizer"):
            self._model.generation_config.pad_token_id = (
                self._processor.tokenizer.eos_token_id
            )
        logger.info("Model loaded")

    # -- Shared inference ------------------------------------------------------

    def _generate(self, messages: list, tools: list = None) -> dict:
        """Core generate + parse_response pipeline.

        Returns the parsed response dict from processor.parse_response():
            {"role": "assistant", "thinking": ..., "content": ..., "tool_calls": ...}

        Falls back to {"content": decoded_text} if parse_response fails.
        """
        if self._model is None:
            self._load_model()

        device = next(self._model.parameters()).device

        chat_kwargs = {"enable_thinking": self.config.enable_thinking}
        if tools:
            chat_kwargs["tools"] = tools

        # Pass max_soft_tokens to control vision token budget
        proc_kwargs = {}
        if self.config.max_soft_tokens != 280:
            proc_kwargs["images_kwargs"] = {
                "max_soft_tokens": self.config.max_soft_tokens
            }

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=proc_kwargs or None,
            **chat_kwargs,
        ).to(device)

        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "repetition_penalty": self.config.repetition_penalty,
        }
        if self.config.do_sample:
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_p"] = self.config.top_p
            gen_kwargs["top_k"] = self.config.top_k
        if self.config.cache_implementation:
            gen_kwargs["cache_implementation"] = self.config.cache_implementation

        with torch.no_grad():
            try:
                output_ids = self._model.generate(**inputs, **gen_kwargs)
            except (IndexError, RuntimeError):
                if "cache_implementation" in gen_kwargs:
                    logger.warning(
                        "cache_implementation='%s' failed, retrying without it",
                        gen_kwargs.pop("cache_implementation"),
                    )
                    output_ids = self._model.generate(**inputs, **gen_kwargs)
                else:
                    raise

        generated = output_ids[0][input_len:]

        # Always use parse_response — it properly separates thinking/content/tool_calls
        raw = self._processor.decode(generated, skip_special_tokens=False)
        try:
            parsed = self._processor.parse_response(raw)
            if isinstance(parsed, dict):
                parsed["_raw"] = raw  # Keep raw for fallback parsing
                return parsed
        except Exception as e:
            logger.debug(f"parse_response failed: {e}")

        # Fallback: plain decode
        return {
            "content": self._processor.decode(
                generated, skip_special_tokens=True
            )
        }

    # -- JSON extraction -------------------------------------------------------

    @staticmethod
    def _repair_json(text: str) -> str:
        """Minimal repair of trivial JSON errors from LLM output.

        SCOPE: Only fixes comma and bracket issues. Do NOT expand this
        method to handle arbitrary malformed JSON.
        """
        s = text.strip()
        s = re.sub(r",{2,}", ",", s)
        s = re.sub(r",\s*([}\]])", r"\1", s)
        s = re.sub(r"(\[|{)\s*,", r"\1", s)
        while s.startswith("[[") and s.endswith("]]"):
            inner = s[1:-1]
            try:
                json.loads(inner)
                s = inner
            except json.JSONDecodeError:
                break
        return s

    def _extract_json(self, text: str) -> Optional[Any]:
        """Extract JSON from model text output."""
        if not text or not text.strip():
            return None

        def _try(s):
            try:
                return json.loads(s)
            except (json.JSONDecodeError, ValueError):
                pass
            try:
                return json.loads(self._repair_json(s))
            except (json.JSONDecodeError, ValueError):
                return None

        # 1. Markdown fence
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            r = _try(m.group(1))
            if r is not None:
                return r

        # 2. Direct parse
        stripped = text.strip()
        if stripped.startswith(("[", "{")):
            r = _try(stripped)
            if r is not None:
                return r

        # 3. Balanced bracket matching
        for sc, ec in [("[", "]"), ("{", "}")]:
            start = text.find(sc)
            if start == -1:
                continue
            depth = 0
            for i in range(start, len(text)):
                if text[i] == sc:
                    depth += 1
                elif text[i] == ec:
                    depth -= 1
                    if depth == 0:
                        r = _try(text[start : i + 1])
                        if r is not None:
                            return r
                        break

        logger.debug(f"No JSON found in: {text[:500]}")
        return None

    # -- Generation parameter properties ---------------------------------------

    @property
    def max_new_tokens(self) -> int:
        return self.config.max_new_tokens

    @max_new_tokens.setter
    def max_new_tokens(self, value: int):
        self.config.max_new_tokens = value

    @property
    def do_sample(self) -> bool:
        return self.config.do_sample

    @do_sample.setter
    def do_sample(self, value: bool):
        self.config.do_sample = value

    @property
    def temperature(self) -> float:
        return self.config.temperature

    @temperature.setter
    def temperature(self, value: float):
        self.config.temperature = value

    @property
    def top_p(self) -> float:
        return self.config.top_p

    @top_p.setter
    def top_p(self, value: float):
        self.config.top_p = value

    @property
    def top_k(self) -> int:
        return self.config.top_k

    @top_k.setter
    def top_k(self, value: int):
        self.config.top_k = value

    @property
    def repetition_penalty(self) -> float:
        return self.config.repetition_penalty

    @repetition_penalty.setter
    def repetition_penalty(self, value: float):
        self.config.repetition_penalty = value

    @property
    def enable_thinking(self) -> bool:
        return self.config.enable_thinking

    @enable_thinking.setter
    def enable_thinking(self, value: bool):
        self.config.enable_thinking = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False


# =============================================================================
# Image config & model
# =============================================================================

# Default max_soft_tokens per operation when not explicitly set by user
# Diagnostic testing showed 280 performs comparably to 560 for detection
# while being faster. OCR benefits from higher resolution.
_OPERATION_SOFT_TOKEN_DEFAULTS = {
    "detect": 280,
    "point": 280,
    "classify": 280,
    "vqa": 280,
    "caption": 280,
    "ocr": 560,
}


class Gemma4ImageModelConfig(Gemma4BaseConfig):
    def __init__(self, d: dict):
        # Set operation-aware max_soft_tokens default before super().__init__
        # so the base class picks it up, but only if user didn't set it
        if "max_soft_tokens" not in d:
            op = d.get("operation", "vqa")
            d["max_soft_tokens"] = _OPERATION_SOFT_TOKEN_DEFAULTS.get(op, 280)

        super().__init__(d)
        self.operation = self.parse_string(d, "operation", default="vqa")
        if self.operation not in IMAGE_OPERATIONS:
            raise ValueError(
                f"Invalid image operation: '{self.operation}'. "
                f"Must be one of {list(IMAGE_OPERATIONS.keys())}"
            )
        self.prompt = self.parse_string(d, "prompt", default=None)
        self.system_prompt = self.parse_string(d, "system_prompt", default=None)


class Gemma4ImageModel(Gemma4BaseModel):
    """FiftyOne zoo model for Gemma 4 image understanding.

    Structured operations (detect, point, classify) use function calling
    via tools= parameter. Text operations (vqa, caption, ocr) use plain
    generation. All outputs go through parse_response for clean separation
    of thinking/content/tool_calls.

    Returns single FiftyOne Label instances (not dicts) for compatibility
    with label_field nesting.
    """

    @property
    def media_type(self) -> str:
        return "image"

    @property
    def operation(self) -> str:
        return self.config.operation

    @operation.setter
    def operation(self, value: str):
        if value not in IMAGE_OPERATIONS:
            raise ValueError(f"Invalid: '{value}'. Must be one of {list(IMAGE_OPERATIONS.keys())}")
        self.config.operation = value

    @property
    def system_prompt(self) -> str:
        return self.config.system_prompt or IMAGE_OPERATIONS[self.config.operation]

    @system_prompt.setter
    def system_prompt(self, value: Optional[str]):
        self.config.system_prompt = value

    @property
    def prompt(self) -> Optional[str]:
        return self.config.prompt

    @prompt.setter
    def prompt(self, value: Optional[str]):
        self.config.prompt = value

    # -- Inference -------------------------------------------------------------

    def _run_inference(self, filepath: str, prompt: str):
        """Run inference on a single image. Returns a FiftyOne Label."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "url": filepath},
                {"type": "text", "text": prompt},
            ]},
        ]

        tools = _OPERATION_TOOLS.get(self.config.operation)
        parsed = self._generate(messages, tools=tools)

        thinking = parsed.get("thinking")

        # 1. Try tool_calls from parse_response (cleanest path)
        tool_calls = parsed.get("tool_calls")
        if tool_calls:
            args = tool_calls[0].get("function", {}).get("arguments", {})
            if args:
                return self._structured_to_label(args, thinking)

        # 2. Try content — might contain tool call text or plain text
        content = parsed.get("content") or ""
        if content.strip().startswith("call:"):
            args = self._parse_tool_call_text(content)
            if args:
                return self._structured_to_label(args, thinking)

        # 3. Fallback: check raw output for tool call that parse_response
        #    failed to extract (e.g. malformed JSON in tool call arguments)
        raw = parsed.get("_raw", "")
        if raw and "call:" in raw and not content:
            # Extract tool call text from raw output
            m = re.search(r"call:\w+(\{.*)", raw, re.DOTALL)
            if m:
                args = self._parse_tool_call_text("call:" + m.group(0).split("call:", 1)[-1])
                if args:
                    return self._structured_to_label(args, thinking)

        # 4. Text content path (text ops, or final fallback)
        return self._text_to_label(content, thinking)

    @staticmethod
    def _parse_tool_call_text(text: str) -> Optional[dict]:
        """Parse a tool call from plain text like 'call:func_name{...}'.

        The model sometimes outputs tool calls as content text rather than
        using the special token format that parse_response expects.
        Handles unquoted keys/values and common format variations.
        """
        m = re.match(r"call:\w+(\{.*)", text.strip(), re.DOTALL)
        if not m:
            return None

        raw_json = m.group(1)

        # Fix common model output issues:
        # 1. bbox_bbox → box_2d (common typo from model)
        fixed = raw_json.replace("bbox_bbox", "box_2d")
        fixed = fixed.replace("bbox_2d", "box_2d")
        # 2. Set-literal bbox {40,10,680,990} → array [40,10,680,990]
        fixed = re.sub(
            r'(box_2d|"box_2d")\s*:\s*\{(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\}',
            r'\1: [\2, \3, \4, \5]',
            fixed,
        )
        # 3. Add quotes around unquoted keys: {key: → {"key":
        fixed = re.sub(r'([{,]\s*)(\w+)\s*:', r'\1"\2":', fixed)
        # 4. Add quotes around unquoted string values (not numbers)
        fixed = re.sub(
            r':\s*([a-zA-Z][a-zA-Z0-9_ ]*?)([,}\]])',
            r': "\1"\2',
            fixed,
        )

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            logger.debug("Failed to parse tool call text: %s", fixed[:500])
            return None

    def _structured_to_label(self, args: dict, thinking):
        """Convert tool call arguments to a FiftyOne Label."""
        op = self.config.operation
        if op == "detect":
            return self._to_detections(args.get("detections", []), thinking)
        if op == "point":
            return self._to_keypoints(args.get("points", []), thinking)
        if op == "classify":
            return self._to_classifications(args.get("labels", []), thinking)
        return fo.Classification(label=str(args))

    def _text_to_label(self, text: str, thinking):
        """Convert plain text content to a FiftyOne Label."""
        op = self.config.operation
        if op in ("vqa", "caption", "ocr"):
            label = fo.Classification(label=text.strip())
            if thinking:
                label["reasoning"] = thinking
            return label

        # Structured op that didn't produce a tool call — try JSON extraction
        data = self._extract_json(text)
        if not data:
            logger.warning("No structured output for %s. Content: %s", op, text[:500])

        if op == "detect":
            return self._to_detections(data, thinking)
        if op == "point":
            return self._to_keypoints(data, thinking)
        if op == "classify":
            return self._to_classifications(data, thinking)
        return fo.Classification(label=text.strip())

    # -- Output converters -----------------------------------------------------

    def _to_detections(self, boxes, reasoning=None) -> fo.Detections:
        """Convert model detection output to fo.Detections.

        Gemma4 native format: {"box_2d": [y1, x1, y2, x2], "label": "..."}
        Coordinates are in 0-1000 scale, converted to FiftyOne's [x, y, w, h]
        in [0, 1] range.
        """
        if not boxes:
            return fo.Detections(detections=[])

        # Unwrap nested lists [[{...}]] → [{...}]
        if isinstance(boxes, list) and boxes and isinstance(boxes[0], list):
            boxes = boxes[0]

        # Unwrap wrapper dicts {"detections": [...]}
        if isinstance(boxes, dict):
            if "box_2d" in boxes or "bbox_2d" in boxes or "bbox" in boxes:
                boxes = [boxes]
            else:
                for v in boxes.values():
                    if isinstance(v, list):
                        boxes = v
                        break
                else:
                    boxes = [boxes]
        elif not isinstance(boxes, list):
            return fo.Detections(detections=[])

        dets = []
        for box in boxes:
            try:
                if isinstance(box, dict):
                    # Try native key first, then fallbacks
                    bbox = (
                        box.get("box_2d")       # Gemma4 native
                        or box.get("bbox_2d")    # our old schema
                        or box.get("bbox_bbox")  # model typo
                        or box.get("bbox")
                        or box.get("bounding_box")
                        or box.get("box")
                    )
                    label = str(box.get("label", box.get("name", "object")))
                elif isinstance(box, list) and len(box) >= 4:
                    bbox, label = box, "object"
                else:
                    continue

                if not bbox:
                    continue

                # Handle bbox as dict {xmin:, ymin:, xmax:, ymax:}
                if isinstance(bbox, dict):
                    x1 = float(bbox.get("xmin", bbox.get("x1", 0)))
                    y1 = float(bbox.get("ymin", bbox.get("y1", 0)))
                    x2 = float(bbox.get("xmax", bbox.get("x2", 0)))
                    y2 = float(bbox.get("ymax", bbox.get("y2", 0)))
                elif isinstance(bbox, list) and len(bbox) >= 4:
                    # Gemma4 native order: [y1, x1, y2, x2]
                    y1, x1, y2, x2 = (float(v) for v in bbox[:4])
                else:
                    continue

                # Normalize to [0, 1]
                mx = max(abs(x1), abs(y1), abs(x2), abs(y2))
                if mx > 1.0:
                    s = 1000.0 if mx <= 1000 else mx
                    x1, y1, x2, y2 = x1 / s, y1 / s, x2 / s, y2 / s

                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue

                det = fo.Detection(label=label, bounding_box=[x1, y1, w, h])
                if reasoning:
                    det["reasoning"] = reasoning
                dets.append(det)
            except Exception as e:
                logger.debug(f"Error processing box {box}: {e}")

        return fo.Detections(detections=dets)

    def _to_keypoints(self, points, reasoning=None) -> fo.Keypoints:
        if not points:
            return fo.Keypoints(keypoints=[])

        if isinstance(points, dict):
            for v in points.values():
                if isinstance(v, list):
                    points = v
                    break
            else:
                return fo.Keypoints(keypoints=[])

        if not isinstance(points, list):
            return fo.Keypoints(keypoints=[])

        if len(points) == 2 and all(isinstance(v, (int, float)) for v in points):
            points = [points]

        kps = []
        for pt in points:
            try:
                if isinstance(pt, list) and len(pt) == 2:
                    # Gemma4 native: [y, x]
                    y_val, x_val = float(pt[0]), float(pt[1])
                    x, y, label = x_val, y_val, "point"
                elif isinstance(pt, dict):
                    coords = pt.get("point_2d") or pt.get("point")
                    if not coords or len(coords) < 2:
                        continue
                    # Gemma4 native: [y, x]
                    y_val, x_val = float(coords[0]), float(coords[1])
                    x, y = x_val, y_val
                    label = str(pt.get("label", "point"))
                else:
                    continue

                mx = max(abs(x), abs(y))
                if mx > 1.0:
                    s = 1000.0 if mx <= 1000 else mx
                    x, y = x / s, y / s

                kp = fo.Keypoint(label=label, points=[[x, y]])
                if reasoning:
                    kp["reasoning"] = reasoning
                kps.append(kp)
            except Exception as e:
                logger.debug(f"Error processing point {pt}: {e}")

        return fo.Keypoints(keypoints=kps)

    def _to_classifications(self, classes, reasoning=None) -> fo.Classifications:
        if not classes:
            return fo.Classifications(classifications=[])

        if isinstance(classes, dict):
            if "label" in classes:
                classes = [classes]
            else:
                for v in classes.values():
                    if isinstance(v, list):
                        classes = v
                        break
                else:
                    classes = [classes]
        elif not isinstance(classes, list):
            return fo.Classifications(classifications=[])

        cls_list = []
        for cls in classes:
            try:
                if isinstance(cls, dict):
                    label = str(cls.get("label", ""))
                elif isinstance(cls, str):
                    label = cls
                else:
                    continue
                if not label:
                    continue
                c = fo.Classification(label=label)
                if reasoning:
                    c["reasoning"] = reasoning
                cls_list.append(c)
            except Exception as e:
                logger.debug(f"Error processing classification {cls}: {e}")

        return fo.Classifications(classifications=cls_list)

    # -- predict / predict_all -------------------------------------------------

    def predict(self, arg, sample=None):
        if isinstance(arg, dict):
            item = arg
        else:
            fp = arg if isinstance(arg, str) else getattr(arg, "inpath", getattr(arg, "path", str(arg)))
            prompt = None
            if sample and "prompt_field" in self._fields:
                fn = self._fields["prompt_field"]
                if sample.has_field(fn):
                    prompt = sample.get_field(fn)
            item = {"filepath": fp, "prompt": prompt, "metadata": None}

        return self.predict_all([item], samples=[sample] if sample else None)[0]

    def predict_all(self, batch: list, samples=None) -> list:
        if not batch:
            return []
        if self._model is None:
            self._load_model()

        results = []
        for item in batch:
            prompt = item.get("prompt") or self.config.prompt
            if not prompt:
                raise ValueError(
                    f"No prompt for '{self.config.operation}'. "
                    "Set model.prompt or pass prompt_field."
                )
            results.append(self._run_inference(item["filepath"], prompt))
        return results


# =============================================================================
# Video config & model
# =============================================================================

class Gemma4VideoModelConfig(Gemma4BaseConfig):
    def __init__(self, d: dict):
        super().__init__(d)
        self.operation = self.parse_string(d, "operation", default="description")
        if self.operation not in VIDEO_OPERATIONS:
            raise ValueError(
                f"Invalid video operation: '{self.operation}'. "
                f"Must be one of {list(VIDEO_OPERATIONS.keys())}"
            )
        self.custom_prompt = self.parse_string(d, "custom_prompt", default=None)
        if self.operation == "custom" and self.custom_prompt is None:
            raise ValueError("custom_prompt required when operation='custom'")
        if self.operation != "custom" and self.custom_prompt is not None:
            raise ValueError("custom_prompt only allowed when operation='custom'")
        if self.model_path not in _VIDEO_CAPABLE_MODELS:
            raise ValueError(
                f"'{self.model_path}' doesn't support video. "
                f"Use one of: {sorted(_VIDEO_CAPABLE_MODELS)}"
            )


class Gemma4VideoModel(Gemma4BaseModel):
    """FiftyOne zoo model for Gemma 4 video understanding."""

    @property
    def media_type(self) -> str:
        return "video"

    @property
    def operation(self) -> str:
        return self.config.operation

    @operation.setter
    def operation(self, value: str):
        if value not in VIDEO_OPERATIONS:
            raise ValueError(f"Invalid: '{value}'. Must be one of {list(VIDEO_OPERATIONS.keys())}")
        self.config.operation = value

    @property
    def prompt(self) -> Optional[str]:
        if self.config.operation == "custom":
            return self.config.custom_prompt
        return VIDEO_OPERATIONS[self.config.operation]["prompt"]

    @prompt.setter
    def prompt(self, value: str):
        if self.config.operation != "custom":
            raise ValueError("Use operation='custom' to set prompt directly.")
        self.config.custom_prompt = value

    @property
    def custom_prompt(self) -> Optional[str]:
        return self.config.custom_prompt if self.config.operation == "custom" else None

    @custom_prompt.setter
    def custom_prompt(self, value: str):
        if self.config.operation != "custom":
            raise ValueError("custom_prompt only allowed when operation='custom'")
        self.config.custom_prompt = value

    # -- Inference -------------------------------------------------------------

    def _run_inference(self, filepath: str, prompt: str) -> str:
        """Run video inference. Returns content text from parse_response."""
        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": filepath},
                {"type": "text", "text": prompt},
            ]},
        ]
        parsed = self._generate(messages)
        return parsed.get("content") or ""

    def _parse_output(self, text: str, sample) -> dict:
        """Parse video text output into FiftyOne labels."""
        if self.config.operation == "description":
            return {"summary": text}
        if self.config.operation == "custom":
            return {"result": text}

        data = self._extract_json(text)

        if self.config.operation == "temporal_localization":
            return self._parse_temporal_only(data, sample)
        if self.config.operation == "tracking":
            return self._parse_tracking_only(data, sample)
        if self.config.operation == "ocr":
            return self._parse_ocr_only(data, sample)
        if self.config.operation == "comprehensive":
            return self._parse_comprehensive(data, sample)

        return {"summary": text}

    # -- Video parsers ---------------------------------------------------------

    def _parse_temporal_only(self, data, sample) -> dict:
        items = data if isinstance(data, list) else (data or {}).get("events", []) or []
        if not items:
            return {"events": fol.TemporalDetections(detections=[])}
        dets = self._parse_temporal_detections(items, sample, "events")
        return {"events": dets or fol.TemporalDetections(detections=[])}

    def _parse_tracking_only(self, data, sample) -> dict:
        items = data if isinstance(data, list) else (data or {}).get("objects", []) or []
        if not items:
            return {"objects": fol.Detections(detections=[])}
        fd = self._parse_frame_detections(items, sample)
        if not fd:
            return {"objects": fol.Detections(detections=[])}
        return {fn: {"objects": d} for fn, d in fd.items()}

    def _parse_ocr_only(self, data, sample) -> dict:
        items = data if isinstance(data, list) else (data or {}).get("text_content", []) or []
        if not items:
            return {"text_content": fol.Detections(detections=[])}
        fd = self._parse_frame_detections(items, sample, text_key="text")
        if not fd:
            return {"text_content": fol.Detections(detections=[])}
        return {fn: {"text_content": d} for fn, d in fd.items()}

    def _parse_comprehensive(self, data, sample) -> dict:
        if not data or not isinstance(data, dict):
            return {"summary": str(data) if data else "No output"}
        labels = {}
        for k, v in data.items():
            if isinstance(v, str):
                labels[k] = v
            elif isinstance(v, dict) and all(isinstance(x, (str, int, float, bool)) for x in v.values()):
                for sk, sv in v.items():
                    labels[f"{k}_{sk}"] = fol.Classification(label=str(sv).capitalize())
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                first = v[0]
                if all(x in first for x in ["start", "end", "description"]):
                    d = self._parse_temporal_detections(v, sample, "events")
                    if d:
                        labels[k] = d
                elif all(x in first for x in ["time", "bbox_2d"]):
                    fd = self._parse_frame_detections(v, sample, text_key="text" if "text" in first else None)
                    for fn, d in fd.items():
                        labels.setdefault(fn, {})[k] = d
        return labels

    def _parse_temporal_detections(self, items, sample, label_type):
        dets = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if label_type == "events":
                start, end = item.get("start", "00:00.00"), item.get("end", "00:00.00")
                label = str(item.get("description", "event")).capitalize()
            elif label_type == "objects":
                start, end = item.get("first_appears", "00:00.00"), item.get("last_appears", "00:00.00")
                label = str(item.get("name", "object")).capitalize()
            else:
                start, end = item.get("start", "00:00.00"), item.get("end", "00:00.00")
                label = str(item.get("text", "text")).capitalize()
            s_sec = self._ts(start)
            e_sec = self._ts(end)
            dets.append(fol.TemporalDetection.from_timestamps([s_sec, e_sec], label=label, sample=sample))
        return fol.TemporalDetections(detections=dets) if dets else None

    def _parse_frame_detections(self, items, sample, text_key=None):
        fps = self._get_fps(sample)
        frames = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            fn = int(self._ts(item.get("time", "00:00.00")) * fps) + 1
            bbox = item.get("bbox_2d", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [max(0, min(1000, c)) for c in bbox[:4]]
            if x2 <= x1 or y2 <= y1:
                continue
            label = item.get("text" if text_key else "label", "")
            det = fol.Detection(label=label, bounding_box=[x1/1000, y1/1000, (x2-x1)/1000, (y2-y1)/1000])
            if text_key:
                det[text_key] = item.get(text_key, "")
            frames.setdefault(fn, fol.Detections(detections=[])).detections.append(det)
        return frames

    @staticmethod
    def _ts(t: str) -> float:
        m = re.match(r"(\d+):(\d+)\.(\d+)", str(t))
        return (int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100.0) if m else 0.0

    @staticmethod
    def _get_fps(sample) -> float:
        if sample:
            meta = getattr(sample, "metadata", None)
            if meta and hasattr(meta, "frame_rate"):
                return meta.frame_rate
        return 30.0

    # -- predict / predict_all -------------------------------------------------

    def predict(self, arg, sample=None):
        if isinstance(arg, dict):
            item = arg
        else:
            fp = arg if isinstance(arg, str) else getattr(arg, "inpath", getattr(arg, "path", str(arg)))
            prompt = None
            if sample and "prompt_field" in self._fields:
                fn = self._fields["prompt_field"]
                if sample.has_field(fn):
                    prompt = sample.get_field(fn)
            item = {"filepath": fp, "prompt": prompt, "metadata": getattr(sample, "metadata", None) if sample else None}
        return self.predict_all([item], samples=[sample] if sample else None)[0]

    def predict_all(self, batch: list, samples=None) -> list:
        if not batch:
            return []
        if self._model is None:
            self._load_model()

        results = []
        for i, item in enumerate(batch):
            sample = samples[i] if samples else None
            needs_meta = self.config.operation in ("comprehensive", "temporal_localization", "tracking", "ocr")
            if needs_meta and not item.get("metadata"):
                raise ValueError(f"'{self.config.operation}' requires metadata. Call dataset.compute_metadata().")

            prompt = item.get("prompt")
            if self.config.operation == "custom" and prompt:
                pass  # use per-sample prompt
            else:
                prompt = self.prompt

            text = self._run_inference(item["filepath"], prompt)
            labels = self._parse_output(text, sample)
            if not any(isinstance(k, int) for k in labels):
                labels["raw"] = text
            results.append(labels)
        return results
