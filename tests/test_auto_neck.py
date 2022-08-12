import unittest


class TestAutoNeck(unittest.TestCase):
    def test_transformers_neck(self):
        import torch

        from video_transformers import AutoNeck

        config = {
            "name": "TransformerNeck",
            "num_features": 256,
            "num_timesteps": 8,
            "transformer_enc_num_heads": 4,
            "transformer_enc_num_layers": 2,
            "transformer_enc_act": "gelu",
            "dropout_p": 0.1,
            "return_mean": True,
        }
        batch_size = 2

        neck = AutoNeck.from_config(config)
        input = torch.randn(batch_size, config["num_timesteps"], config["num_features"])
        output = neck(input)
        self.assertEqual(output.shape, (batch_size, neck.num_features))

    def test_lstm_neck(self):
        import torch

        from video_transformers import AutoNeck

        config = {
            "name": "LSTMNeck",
            "num_features": 256,
            "num_timesteps": 8,
            "hidden_size": 128,
            "num_layers": 2,
            "return_last": True,
        }
        batch_size = 2

        neck = AutoNeck.from_config(config)
        input = torch.randn(batch_size, config["num_timesteps"], config["num_features"])
        output = neck(input)
        self.assertEqual(output.shape, (batch_size, config["hidden_size"]))

    def test_gru_neck(self):
        import torch

        from video_transformers import AutoNeck

        config = {
            "name": "GRUNeck",
            "num_features": 256,
            "num_timesteps": 8,
            "hidden_size": 128,
            "num_layers": 2,
            "return_last": True,
        }
        batch_size = 2

        neck = AutoNeck.from_config(config)
        input = torch.randn(batch_size, config["num_timesteps"], config["num_features"])
        output = neck(input)
        self.assertEqual(output.shape, (batch_size, config["hidden_size"]))


if __name__ == "__main__":
    unittest.main()
