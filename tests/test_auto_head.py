import unittest


class TestAutoHead(unittest.TestCase):
    def test_linear_head(self):
        import torch

        from video_transformers import AutoHead

        config = {
            "name": "LinearHead",
            "hidden_size": 256,
            "num_classes": 10,
            "dropout_p": 0.1,
        }
        batch_size = 2

        head = AutoHead.from_config(config)
        input = torch.randn(batch_size, config["hidden_size"])
        output = head(input)
        self.assertEqual(output.shape, (batch_size, config["num_classes"]))

    def test_from_transformers(self):
        from video_transformers import AutoHead

        linear_head = AutoHead.from_transformers("facebook/timesformer-base-finetuned-k400")
        assert linear_head.num_classes == 400
        linear_head = AutoHead.from_transformers("facebook/timesformer-base-finetuned-k600")
        assert linear_head.num_classes == 600
        linear_head = AutoHead.from_transformers("facebook/timesformer-hr-finetuned-k400")
        assert linear_head.num_classes == 400
        linear_head = AutoHead.from_transformers("facebook/timesformer-hr-finetuned-k600")
        assert linear_head.num_classes == 600
        linear_head = AutoHead.from_transformers("facebook/timesformer-base-finetuned-ssv2")
        assert linear_head.num_classes == 174
        linear_head = AutoHead.from_transformers("facebook/timesformer-hr-finetuned-ssv2")
        assert linear_head.num_classes == 174


if __name__ == "__main__":
    unittest.main()
