import unittest


class TestAutoBackbone(unittest.TestCase):
    def test_transformers_backbone(self):
        import torch

        from video_transformers import AutoBackbone

        config = {
            "framework": {"name": "timm"},
            "type": "2d_backbone",
            "model_name": "mobilevitv2_100",
            "num_timesteps": 8,
        }
        batch_size = 2

        backbone = AutoBackbone.from_config(config)
        input = torch.randn(batch_size, 3, config["num_timesteps"], 224, 224)
        output = backbone(input)
        self.assertEqual(output.shape, (batch_size, config["num_timesteps"], backbone.num_features))

    def test_timm_backbone(self):
        import torch

        from video_transformers import AutoBackbone

        config = {
            "framework": {"name": "transformers"},
            "type": "2d_backbone",
            "model_name": "microsoft/cvt-13",
            "num_timesteps": 8,
        }
        batch_size = 2

        backbone = AutoBackbone.from_config(config)
        input = torch.randn(batch_size, 3, config["num_timesteps"], 224, 224)
        output = backbone(input)
        self.assertEqual(output.shape, (batch_size, config["num_timesteps"], backbone.num_features))


if __name__ == "__main__":
    unittest.main()
