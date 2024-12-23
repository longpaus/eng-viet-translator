import torch
from torch import nn
from utils import clones
from layers import EncoderLayer


class Encoder(nn.Module):

    def __init__(self, d_model:int, h:int, d_ff:int, N:int):
        """
        Initializes the Encoder module.

        Args:
            d_model (int): The dimensionality of the model.
            h (int): The number of attention heads.
            d_ff (int): The dimensionality of the feedforward network.
            N (int): The number of encoder layers.
        """
        super(Encoder, self).__init__()
        self.sublayers = clones(EncoderLayer(d_model, h, d_ff), N)

    def forward(self, x, mask):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): The input tensor. Shape: (batch_size, seq_len, d_model)
            mask (torch.Tensor): The attention mask tensor for padding.

        Returns:
            torch.Tensor: The encoded output tensor. Shape: (batch_size, seq_len, d_model)
        """
        for sublayer in self.sublayers:
            x = sublayer(x, mask)
        return x
