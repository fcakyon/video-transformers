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

    def test_from_transformers(self):
        from video_transformers import AutoBackbone

        backbone = AutoBackbone.from_transformers("facebook/timesformer-base-finetuned-k400")
        assert backbone.model_name == "facebook/timesformer-base-finetuned-k400"
        backbone = AutoBackbone.from_transformers("facebook/timesformer-base-finetuned-k600")
        assert backbone.model_name == "facebook/timesformer-base-finetuned-k600"
        backbone = AutoBackbone.from_transformers("facebook/timesformer-hr-finetuned-k400")
        assert backbone.model_name == "facebook/timesformer-hr-finetuned-k400"
        backbone = AutoBackbone.from_transformers("facebook/timesformer-hr-finetuned-k600")
        assert backbone.model_name == "facebook/timesformer-hr-finetuned-k600"
        backbone = AutoBackbone.from_transformers("facebook/timesformer-base-finetuned-ssv2")
        assert backbone.model_name == "facebook/timesformer-base-finetuned-ssv2"
        backbone = AutoBackbone.from_transformers("facebook/timesformer-hr-finetuned-ssv2")
        assert backbone.model_name == "facebook/timesformer-hr-finetuned-ssv2"


if __name__ == "__main__":
    unittest.main()
