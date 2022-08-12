import unittest


class TestBackbone(unittest.TestCase):
    def test_transformers_backbone(self):
        import torch

        from video_transformers.backbones.transformers import TransformersBackbone

        config = {"model_name": "microsoft/cvt-13"}
        batch_size = 2

        backbone = TransformersBackbone(model_name=config["model_name"], num_unfrozen_stages=0)
        self.assertEqual(backbone.num_trainable_params, 0)

        backbone = TransformersBackbone(model_name=config["model_name"], num_unfrozen_stages=-1)
        self.assertNotEqual(backbone.num_trainable_params, 0)

        input = torch.randn(batch_size, 3, 224, 224)
        output = backbone(input)
        self.assertEqual(output.shape, (batch_size, backbone.num_features))

    def test_timm_backbone(self):
        import torch

        from video_transformers.backbones.timm import TimmBackbone

        config = {"model_name": "mobilevitv2_100"}
        batch_size = 2

        backbone = TimmBackbone(model_name=config["model_name"], num_unfrozen_stages=0)
        self.assertEqual(backbone.num_trainable_params, 0)

        backbone = TimmBackbone(model_name=config["model_name"], num_unfrozen_stages=-1)
        self.assertNotEqual(backbone.num_trainable_params, 0)

        input = torch.randn(batch_size, 3, 224, 224)
        output = backbone(input)
        self.assertEqual(output.shape, (batch_size, backbone.num_features))


if __name__ == "__main__":
    unittest.main()
