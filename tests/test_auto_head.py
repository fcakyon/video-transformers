import unittest


class TestAutoHead(unittest.TestCase):
    def test_liear_head(self):
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


if __name__ == "__main__":
    unittest.main()
