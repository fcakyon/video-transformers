import unittest


class TestOnnx(unittest.TestCase):
    def test_onnx_export(self):
        from video_transformers import VideoModel

        config = {
            "backbone": {
                "name": "TransformersBackbone",
                "framework": {"name": "transformers", "version": "4.21.1"},
                "mean": [0.485, 0.456, 0.406],
                "model_name": "microsoft/cvt-13",
                "num_features": 384,
                "num_total_params": 19611712,
                "num_trainable_params": 18536448,
                "std": [0.229, 0.224, 0.225],
                "type": "2d_backbone",
            },
            "head": {"name": "LinearHead", "dropout_p": 0.0, "hidden_size": 384, "num_classes": 6},
            "neck": {
                "name": "TransformerNeck",
                "dropout_p": 0.1,
                "num_features": 384,
                "num_timesteps": 8,
                "transformer_enc_act": "gelu",
                "transformer_enc_num_heads": 4,
                "transformer_enc_num_layers": 2,
                "return_mean": True,
            },
            "preprocessor": {
                "means": [0.485, 0.456, 0.406],
                "stds": [0.229, 0.224, 0.225],
                "min_short_side": 256,
                "input_size": 224,
                "num_timesteps": 8,
            },
            "labels": ["BodyWeightSquats", "JumpRope", "Lunges", "PullUps", "PushUps", "WallPushups"],
            "task": "single_label_classification",
        }

        model = VideoModel.from_config(config)

        model.to_onnx()

    def test_quantized_onnx_export(self):
        from video_transformers import VideoModel

        config = {
            "backbone": {
                "name": "TransformersBackbone",
                "framework": {"name": "transformers", "version": "4.21.1"},
                "mean": [0.485, 0.456, 0.406],
                "model_name": "microsoft/cvt-13",
                "num_features": 384,
                "num_total_params": 19611712,
                "num_trainable_params": 18536448,
                "std": [0.229, 0.224, 0.225],
                "type": "2d_backbone",
            },
            "head": {"name": "LinearHead", "dropout_p": 0.0, "hidden_size": 384, "num_classes": 6},
            "neck": {
                "name": "TransformerNeck",
                "dropout_p": 0.1,
                "num_features": 384,
                "num_timesteps": 8,
                "transformer_enc_act": "gelu",
                "transformer_enc_num_heads": 4,
                "transformer_enc_num_layers": 2,
                "return_mean": True,
            },
            "preprocessor": {
                "means": [0.485, 0.456, 0.406],
                "stds": [0.229, 0.224, 0.225],
                "min_short_side": 256,
                "input_size": 224,
                "num_timesteps": 8,
            },
            "labels": ["BodyWeightSquats", "JumpRope", "Lunges", "PullUps", "PushUps", "WallPushups"],
            "task": "single_label_classification",
        }

        model = VideoModel.from_config(config)

        model.to_onnx(quantize=True)


if __name__ == "__main__":
    unittest.main()
