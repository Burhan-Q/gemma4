"""Gemma 4 FiftyOne Zoo Model plugin.

Provides image and video understanding via Google Gemma 4 vision-language models.

Image operations:  detect, point, classify, vqa, caption, ocr
Video operations:  description, temporal_localization, tracking, ocr,
                   comprehensive, custom

Usage:
    import fiftyone.zoo as foz

    # Image model
    model = foz.load_zoo_model(
        "google/gemma-4-E4B-it",
        media_type="image",
        operation="detect",
        prompt="Detect all people in the image",
    )
    dataset.apply_model(model, label_field="detections")

    # Video model (E2B/E4B only)
    model = foz.load_zoo_model(
        "google/gemma-4-E4B-it",
        media_type="video",
        operation="description",
    )
    dataset.apply_model(model, label_field="description")
"""

import logging
from typing import Any

from huggingface_hub import snapshot_download
from fiftyone.operators import types

from .zoo import (
    IMAGE_OPERATIONS,
    VIDEO_OPERATIONS,
    Gemma4ImageModel,
    Gemma4ImageModelConfig,
    Gemma4VideoModel,
    Gemma4VideoModelConfig,
)

logger = logging.getLogger(__name__)


def download_model(model_name: str, model_path: str) -> None:
    """Download the Gemma 4 model from HuggingFace.

    Args:
        model_name: HuggingFace repo ID (e.g. "google/gemma-4-E4B-it")
        model_path: Local directory to download into
    """
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(
    model_name: str | None = None, model_path: str | None = None, **kwargs: Any
) -> Gemma4ImageModel | Gemma4VideoModel:
    """Load a Gemma 4 model for use with FiftyOne.

    Args:
        model_name: Model name (unused, kept for zoo compatibility)
        model_path: HuggingFace model ID or local path to model files.
            Defaults to "google/gemma-4-E4B-it".
        **kwargs: Config parameters. Key ones:

            media_type (str): "image" or "video". Default: "image".

            --- Image params ---
            operation (str):  One of detect, point, classify, vqa, caption, ocr.
                              Default: "vqa".
            prompt (str):     User instruction for the image operation.
            system_prompt (str): Optional override for the default system prompt.

            --- Video params ---
            operation (str):  One of description, temporal_localization, tracking,
                              ocr, comprehensive, custom. Default: "description".
            custom_prompt (str): Required when operation="custom".

            --- Shared generation params ---
            max_new_tokens (int): Default 2048.
            do_sample (bool):     Default True.
            temperature (float):  Default 1.0.
            top_p (float):        Default 0.95.
            top_k (int):          Default 64.
            repetition_penalty (float): Default 1.0.
            enable_thinking (bool): Default False.
            max_soft_tokens (int): Vision token budget per image.
                              One of 70, 140, 280, 560, 1120.
                              Default varies by operation: 560 for detect/point/ocr,
                              280 for vqa/caption/classify. User override respected.
            cache_implementation (str): KV cache strategy for generate().
                              "static" pre-allocates cache (used in official examples).
                              Default: None (transformers default).

    Returns:
        Gemma4ImageModel or Gemma4VideoModel
    """
    if model_path is None:
        model_path = "google/gemma-4-E4B-it"

    media_type: str = kwargs.pop("media_type", "image")

    config_dict: dict[str, Any] = {"model_path": model_path}
    config_dict.update(kwargs)

    if media_type == "video":
        config = Gemma4VideoModelConfig(config_dict)
        return Gemma4VideoModel(config)

    config = Gemma4ImageModelConfig(config_dict)
    return Gemma4ImageModel(config)


def resolve_input(model_name: str, ctx: Any) -> types.Property:
    """Define FiftyOne operator UI inputs for this model.

    Args:
        model_name: The name of the model
        ctx:        An ExecutionContext

    Returns:
        fiftyone.operators.types.Property
    """
    inputs = types.Object()

    # -------------------------------------------------------------------------
    # Media type
    # -------------------------------------------------------------------------
    inputs.enum(
        "media_type",
        values=["image", "video"],
        default="image",
        label="Media Type",
        description="Whether to process images or videos",
    )

    # -------------------------------------------------------------------------
    # Operation
    # -------------------------------------------------------------------------
    all_operations = list(IMAGE_OPERATIONS.keys()) + [
        k for k in VIDEO_OPERATIONS.keys() if k != "custom"
    ]
    inputs.enum(
        "operation",
        values=all_operations,
        default="vqa",
        label="Operation",
        description=(
            "Image ops: detect, point, classify, vqa, caption, ocr. "
            "Video ops: description, temporal_localization, tracking, "
            "ocr, comprehensive."
        ),
    )

    # -------------------------------------------------------------------------
    # Image-only parameters
    # -------------------------------------------------------------------------
    inputs.str(
        "prompt",
        default=None,
        required=False,
        label="Prompt",
        description=(
            "User instruction for image operations (detect, point, classify, vqa)"
        ),
    )

    inputs.str(
        "system_prompt",
        default=None,
        required=False,
        label="System Prompt Override",
        description=(
            "Optional: override the default system prompt for image operations"
        ),
    )

    # -------------------------------------------------------------------------
    # Video-only parameters
    # -------------------------------------------------------------------------
    inputs.str(
        "custom_prompt",
        default=None,
        required=False,
        label="Custom Prompt (video only)",
        description="Required when operation is 'custom' for video",
    )

    # -------------------------------------------------------------------------
    # Shared generation parameters
    # -------------------------------------------------------------------------
    inputs.int(
        "max_new_tokens",
        default=512,
        label="Max New Tokens",
        description="Maximum tokens to generate in the response",
    )

    inputs.bool(
        "do_sample",
        default=True,
        label="Use Sampling",
        description="Use sampling (True) vs greedy decoding (False)",
    )

    inputs.float(
        "temperature",
        default=1.0,
        label="Temperature",
        description=("Sampling temperature (only used when Use Sampling is True)"),
    )

    inputs.float(
        "top_p",
        default=0.95,
        label="Top-p",
        description=(
            "Nucleus sampling threshold (only used when Use Sampling is True)"
        ),
    )

    inputs.int(
        "top_k",
        default=64,
        label="Top-k",
        description=("Top-k sampling parameter (only used when Use Sampling is True)"),
    )

    inputs.float(
        "repetition_penalty",
        default=1.0,
        label="Repetition Penalty",
        description=(
            "Penalizes tokens that have already appeared in the output, "
            "reducing repetition."
        ),
    )

    inputs.bool(
        "enable_thinking",
        default=False,
        label="Enable Thinking",
        description=(
            "Enable Gemma 4 reasoning mode. The model will show "
            "step-by-step reasoning before the final answer."
        ),
    )

    inputs.int(
        "max_soft_tokens",
        default=280,
        label="Vision Token Budget",
        description=(
            "Tokens per image (70, 140, 280, 560, 1120). "
            "Default varies by operation: 560 for detect/point/ocr, "
            "280 for vqa/caption/classify."
        ),
    )

    inputs.str(
        "cache_implementation",
        default=None,
        required=False,
        label="Cache Implementation",
        description=(
            "KV cache strategy for generate(). "
            "'static' pre-allocates cache (used in official Gemma4 examples). "
            "Leave empty for default."
        ),
    )

    return types.Property(inputs)
