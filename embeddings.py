import torch
from torch import nn


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model:int, vocab_size:int, max_seq_len:int, pad_token:int=0):
        """
        A module that combines word embeddings with positional encodings.
        Args:
			d_model (int): The dimensionality of the word embeddings and positional encodings.
			vocab_size (int): The size of the vocabulary (number of unique tokens).
			max_seq_len (int, optional): The maximum sequence length.
        """
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token)
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.register_buffer('pos_encoding', self._get_pos_encoding())

    def _get_pos_encoding(self):
        """
        Generates the positional encoding matrix.

        The positional encoding is a matrix that encodes the position of each token in a sequence.
        It is added to the word embeddings to give the model information about the position of each token.

        Returns:
            torch.Tensor: The positional encoding matrix of shape (1, max_seq_len, d_model).
        """
        positions = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        dimensions = torch.arange(0, self.d_model, dtype=torch.float)
        denominators = torch.pow(10000, 2 * dimensions / self.d_model)
        pe = positions / denominators
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        """
        Forward pass of the PositionalEmbedding layer.

        Applies the word embedding to the input indices and adds the positional encoding.

        Args:
            x (torch.Tensor): Input tensor of token indices with shape (batch_size, seq_len). Assume it is padded.

        Returns:
            torch.Tensor: Tensor of word embeddings with positional encodings added.
                        Shape: (batch_size, max_seq_len, d_model)
        """
        _, seq_len = x.shape
        if seq_len > self.max_seq_len:  # explicitly check out-of-bound slicing
            raise RuntimeError("Sequence length exceeds the maximum allowed limit")
        
        embeddings = self.embedding(x)  # (batch_size,seq_len)
        encodings = embeddings + self.pos_encoding[:,:seq_len,:]
        return encodings
