import unittest


class TestVideoClassificationModel(unittest.TestCase):
    def test_transformers_backbone(self):
        import torch

        from video_transformers import VideoClassificationModel

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
            "preprocess_means": [0.485, 0.456, 0.406],
            "preprocess_tds": [0.229, 0.224, 0.225],
            "preprocess_min_short_side_scale": 256,
            "preprocess_input_size": 224,
            "num_timesteps": 8,
            "labels": ["BodyWeightSquats", "JumpRope", "Lunges", "PullUps", "PushUps", "WallPushups"],
        }
        batch_size = 2

        model = VideoClassificationModel.from_config(config)

        input = torch.randn(batch_size, 3, config["num_timesteps"], 224, 224)
        output = model(input)
        self.assertEqual(output.shape, (batch_size, model.head.num_classes))


if __name__ == "__main__":
    unittest.main()
