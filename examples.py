"""Minimal reproducible examples for each Gemma4 task type.

Usage:
    uv run examples.py [task]

Where [task] is one of:
    vqa, caption, ocr, detect, point, classify,
    video_description, video_custom
    all  (runs all tasks)

Each example uses the smallest possible FiftyOne Zoo dataset slice.
"""

import sys
import fiftyone as fo
import fiftyone.zoo as foz

# Import from local package — add parent dir so 'gemma4' is importable
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from gemma4 import load_model

MODEL_ID = "google/gemma-4-E4B-it"


def run_vqa():
    """Visual Question Answering — returns fo.Classification with text answer."""
    ds = foz.load_zoo_dataset("quickstart", max_samples=2, dataset_name="gemma4_vqa_example")
    model = load_model(model_path=MODEL_ID, operation="vqa")
    model.prompt = "What is the main subject of this image?"
    ds.apply_model(model, label_field="gemma4_vqa")

    # NOTE: do not truncate results when viewing — full output is important
    for s in ds:
        print(f"  {s.filepath.split('/')[-1]}: {s.gemma4_vqa.label}")

    ds.delete()
    print("  [OK] VQA\n")


def run_caption():
    """Caption — returns fo.Classification with descriptive text."""
    ds = foz.load_zoo_dataset("quickstart", max_samples=2, dataset_name="gemma4_caption_example")
    model = load_model(model_path=MODEL_ID, operation="caption")
    model.prompt = "Describe this image in one sentence."
    ds.apply_model(model, label_field="gemma4_caption")

    # NOTE: do not truncate results when viewing — full output is important
    for s in ds:
        print(f"  {s.filepath.split('/')[-1]}: {s.gemma4_caption.label}")

    ds.delete()
    print("  [OK] Caption\n")


def run_ocr():
    """OCR — returns fo.Classification with extracted text."""
    from fiftyone.utils.huggingface import load_from_hub

    ds = load_from_hub(
        "Voxel51/visual_ai_at_neurips2025",
        max_samples=2,
        dataset_name="gemma4_ocr_example",
    )
    model = load_model(model_path=MODEL_ID, operation="ocr")
    model.prompt = "Extract all visible text from this image."
    ds.apply_model(model, label_field="gemma4_ocr")

    # NOTE: do not truncate results when viewing — full output is important
    for s in ds:
        text = s.gemma4_ocr.label if s.gemma4_ocr else "N/A"
        print(f"  {s.filepath.split('/')[-1]}:")
        print(f"    {text}")
        print()

    ds.delete()
    print("  [OK] OCR\n")


def run_detect():
    """Detection — returns fo.Detections with bounding boxes."""
    ds = foz.load_zoo_dataset("quickstart", max_samples=2, dataset_name="gemma4_detect_example")
    model = load_model(model_path=MODEL_ID, operation="detect")
    model.prompt = "Detect all objects."
    ds.apply_model(model, label_field="gemma4_detect")

    # NOTE: do not truncate results when viewing — full output is important
    for s in ds:
        dets = s.gemma4_detect
        n = len(dets.detections) if dets else 0
        print(f"  {s.filepath.split('/')[-1]}: {n} detections")
        if dets:
            for d in dets.detections:
                # bounding_box is [x, y, w, h] in [0, 1] normalized coords
                box = [round(v, 3) for v in d.bounding_box]
                print(f"    label={d.label}  bbox={box}")
        print()

    ds.delete()
    print("  [OK] Detect\n")


def run_point():
    """Keypoint detection — returns fo.Keypoints."""
    ds = foz.load_zoo_dataset("quickstart", max_samples=2, dataset_name="gemma4_point_example")
    model = load_model(model_path=MODEL_ID, operation="point")
    model.prompt = "Point to the center of each animal in this image."
    ds.apply_model(model, label_field="gemma4_point")

    # NOTE: do not truncate results when viewing — full output is important
    for s in ds:
        kps = s.gemma4_point
        n = len(kps.keypoints) if kps else 0
        print(f"  {s.filepath.split('/')[-1]}: {n} keypoints")
        if kps:
            for k in kps.keypoints:
                # points is [[x, y]] in [0, 1] normalized coords
                pts = [[round(v, 3) for v in pt] for pt in k.points]
                print(f"    label={k.label}  points={pts}")
        print()

    ds.delete()
    print("  [OK] Point\n")


def run_classify():
    """Classification — returns fo.Classifications."""
    ds = foz.load_zoo_dataset("quickstart", max_samples=2, dataset_name="gemma4_classify_example")
    model = load_model(model_path=MODEL_ID, operation="classify")
    model.prompt = "Classify this image."
    ds.apply_model(model, label_field="gemma4_classify")

    # NOTE: do not truncate results when viewing — full output is important
    for s in ds:
        cls = s.gemma4_classify
        labels = [c.label for c in cls.classifications] if cls else []
        print(f"  {s.filepath.split('/')[-1]}: {labels}")

    ds.delete()
    print("  [OK] Classify\n")


def _check_ffprobe():
    """Check if ffprobe is available (required for video processing)."""
    import shutil
    if not shutil.which("ffprobe"):
        print("  [SKIP] ffprobe not found. Install ffmpeg: brew install ffmpeg")
        return False
    return True


def run_video_description():
    """Video description — returns dict with 'summary' string."""
    if not _check_ffprobe():
        return

    ds = foz.load_zoo_dataset("quickstart-video", max_samples=1, dataset_name="gemma4_vdesc_example")
    model = load_model(model_path=MODEL_ID, media_type="video", operation="description")
    ds.apply_model(model, label_field="gemma4_desc")

    for s in ds:
        for f in s.field_names:
            if "gemma4" in f:
                print(f"  {f}: {str(s[f])[:150]}")

    ds.delete()
    print("  [OK] Video Description\n")


def run_video_custom():
    """Video custom prompt — returns dict with 'result' string."""
    if not _check_ffprobe():
        return

    ds = foz.load_zoo_dataset("quickstart-video", max_samples=1, dataset_name="gemma4_vcustom_example")
    model = load_model(
        model_path=MODEL_ID,
        media_type="video",
        operation="custom",
        custom_prompt="How many people appear in this video?",
    )
    ds.apply_model(model, label_field="gemma4_custom")

    for s in ds:
        for f in s.field_names:
            if "gemma4" in f:
                print(f"  {f}: {str(s[f])[:150]}")

    ds.delete()
    print("  [OK] Video Custom\n")


TASKS = {
    "vqa": run_vqa,
    "caption": run_caption,
    "ocr": run_ocr,
    "detect": run_detect,
    "point": run_point,
    "classify": run_classify,
    "video_description": run_video_description,
    "video_custom": run_video_custom,
}


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "all"

    if task == "all":
        for name, fn in TASKS.items():
            print(f"=== {name} ===")
            try:
                fn()
            except Exception as e:
                print(f"  [FAIL] {name}: {e}\n")
    elif task in TASKS:
        print(f"=== {task} ===")
        TASKS[task]()
    else:
        print(f"Unknown task: {task}")
        print(f"Available: {', '.join(TASKS.keys())}, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
