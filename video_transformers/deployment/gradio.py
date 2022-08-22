from pathlib import Path
from typing import List

from video_transformers.templates import generate_gradio_app


def export_gradio_app(
    model_url: str,
    examples: List[str],
    author_username: str = None,
    export_dir: str = "runs/exports/",
    export_filename: str = "app.py",
) -> str:
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    export_path = Path(export_dir) / export_filename
    # save as gradio app
    with open(export_path, "w") as f:
        f.write(generate_gradio_app(model_url, examples, author_username))
