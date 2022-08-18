import tempfile
from pathlib import Path
from typing import Optional

import torch

from video_transformers.utils.imports import check_requirements


def export(
    model,
    quantize: bool = False,
    opset_version: int = 12,
    export_dir: str = "runs/exports/",
    export_filename: str = "model.onnx",
):
    """
    Exports a model to ONNX format.

    Args:
        model (VideoClassificationModel): The model to export.
        quantize (bool): Whether to quantize the model.
        opset_version (int): The ONNX opset version.
        export_dir (str): The directory to export the model to.
        export_filename (str): The filename to export the model to.
    """
    check_requirements(["torch", "onnx"])

    import onnx

    model.eval()

    Path(export_dir).mkdir(parents=True, exist_ok=True)
    export_path = Path(export_dir) / export_filename
    print(f"Export Path: {export_path}")
    dynamic_axes = {
        "input_data": {0: "batch", 1: "timestamp", 3: "height", 4: "width"},  # write axis names
        "preds": {0: "batch"},
    }
    input_names = ["input_data"]
    output_names = ["preds"]

    if quantize:
        check_requirements(["onnxruntime"])

        from onnxruntime.quantization import quantize_dynamic

        export_filename = Path(export_path).stem + f"_quantized.{Path(export_path).suffix}"

        target_model_path = Path(export_path).parent / export_filename

        with tempfile.NamedTemporaryFile(suffix=".onnx") as temp:
            torch.onnx.export(
                model,
                f=temp.name,
                args=model.example_input_array,
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                export_params=True,
            )
            quantize_dynamic(temp.name, target_model_path)
    else:
        target_model_path = export_path
        torch.onnx.export(
            model,
            f=target_model_path,
            args=model.example_input_array,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_params=True,
        )

    onnx_model = onnx.load(target_model_path)
    meta = onnx_model.metadata_props.add()
    meta.key = "labels"
    meta.value = "\n".join(model.labels)
    onnx.save(onnx_model, target_model_path)
    print("Model saved")
