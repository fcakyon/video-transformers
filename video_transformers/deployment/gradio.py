from pathlib import Path
from typing import List

from video_transformers.templates import generate_gradio_app


def export_gradio_app(
    model,
    examples: List[str],
    author_username: str = None,
    export_dir: str = "runs/exports/",
    export_filename: str = "app.py",
) -> str:
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    app_path = Path(export_dir) / export_filename
    model_dir = str(Path(export_dir) / "checkpoint")
    # save model
    model.save_pretrained(model_dir, config=model.config)
    # save as gradio app
    with open(app_path, "w") as f:
        f.write(generate_gradio_app(model_dir, examples, author_username))
