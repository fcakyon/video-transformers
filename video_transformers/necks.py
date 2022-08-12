import math
from typing import Dict

import torch
from torch import nn

from video_transformers.utils.extra import class_to_config


class BaseNeck(nn.Module):
    @property
    def config(self) -> Dict:
        return class_to_config(self)


class LSTMNeck(BaseNeck):
    """
        (BxTxF)
           ↓
         LSTM
           ↓
    (BxF) or (BxTxF)
    """

    def __init__(self, num_features, hidden_size, num_layers, return_last=True):
        """
        Create a LSTMNeck.

        Args:
            num_features: Number of input features.
            hidden_size: Number of hidden units.
            num_layers: Number of layers.
            return_last: If True, return the last hidden state of the LSTM.
        """
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_last = return_last

        self.lstm = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F1: num_features, F2: hidden_size
        # x: BxTxF1
        x, _ = self.lstm(x)  # x: BxTxF2
        if self.return_last:
            x = x[:, -1, :]  # x: BxF2
        return x


class GRUNeck(BaseNeck):
    """
        (BxTxF)
           ↓
          GRU
           ↓
    (BxF) or (BxTxF)
    """

    def __init__(self, num_features: int, hidden_size: int, num_layers: int, return_last: bool = True):
        """
        Create a GRUNeck.

        Args:
            num_features: Number of input features.
            hidden_size: Number of hidden units.
            num_layers: Number of layers.
            return_last: If True, return the last hidden state of the GRU.
        """
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_last = return_last

        self.gru = nn.GRU(num_features, hidden_size, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F1: num_features, F2: hidden_size
        # x: BxTxF1
        x, _ = self.gru(x)  # x: BxTxF2
        if self.return_last:
            x = x[:, -1, :]  # x: BxF2
        return x


class PostionalEncoder(nn.Module):
    """
         (BxTxF)
            ↓
    PostionalEncoder
            ↓
         (BxTxF)
    """

    def __init__(self, num_features: int, dropout_p: float = 0.0, num_timesteps: int = 30):
        super(PostionalEncoder, self).__init__()

        self.num_features = num_features
        self.dropout_p = dropout_p
        self.num_timesteps = num_timesteps

        self.dropout = nn.Dropout(dropout_p) if dropout_p != 0 else None

        self.scale_constat = torch.sqrt(torch.tensor(self.num_features))

        position_encodings = torch.zeros(self.num_timesteps, self.num_features)  # pe: TxF
        for time_ind in range(self.num_timesteps):
            for feat_ind in range(0, self.num_features, 2):
                sin_input = time_ind / (10000 ** ((2 * feat_ind) / self.num_features))
                cos_input = time_ind / (10000 ** ((2 * (feat_ind + 1)) / self.num_features))
                position_encodings[time_ind, feat_ind] = torch.sin(torch.tensor(sin_input))
                position_encodings[time_ind, feat_ind + 1] = torch.cos(torch.tensor(cos_input))

        self.position_encodings = position_encodings

    def add_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        # x: BxTxF
        self.position_encodings = self.position_encodings.to(x.device)
        batch_size = x.size(0)
        x = x + self.position_encodings.repeat(batch_size, 1, 1)  # pe: BxTxF
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: BxTxF

        # end of section 3.4 of 'Attention is All You Need' paper
        x = x * self.scale_constat  # x: BxTxF

        x = self.add_positional_encoding(x)  # x: BxTxF
        if self.dropout is not None:
            x = self.dropout(x)  # x: BxTxF
        return x


class TransformerNeck(BaseNeck):
    """
        (BxTxF)
           ↓
      Transformer
           ↓
    (BxF) or (BxTxF)
    """

    def __init__(
        self,
        num_features: int,
        num_timesteps: int,
        transformer_enc_num_heads: int = 4,
        transformer_enc_num_layers: int = 2,
        transformer_enc_act: int = "gelu",
        dropout_p: int = 0.0,
        return_mean: bool = True,
    ):
        """
        Create a TransformerNeck.

        Args:
            num_features: number of input features
            num_timesteps: number of timesteps
            transformer_enc_num_heads: number of heads in the transformer encoder
            transformer_enc_num_layers: number of layers in the transformer encoder
            transformer_enc_act: activation function for the transformer encoder
            dropout_p: dropout probability
            return_mean: return the mean of the transformed features
        """

        super(TransformerNeck, self).__init__()

        self.num_features = num_features
        self.num_timesteps = num_timesteps
        self.transformer_enc_num_heads = transformer_enc_num_heads
        self.transformer_enc_num_layers = transformer_enc_num_layers
        self.transformer_enc_act = transformer_enc_act
        self.dropout_p = dropout_p
        self.return_mean = return_mean

        self.positional_encoder = PostionalEncoder(num_features, dropout_p, num_timesteps)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=transformer_enc_num_heads,
            activation=transformer_enc_act,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=transformer_enc_num_layers)

    def forward(self, x):
        # F: num_features
        # x: BxTxF
        x = self.positional_encoder(x)  # x: BxTxF
        x = self.transformer_encoder(x)  # x: BxTxF
        if self.return_mean:
            x = x.mean(dim=1)  # x: BxF
        return x
