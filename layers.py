import torch
from torch import nn
from utils import attention_score, clones

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sublayer_connection(x: torch.Tensor, sublayer_out: torch.Tensor, dropout_p: float=0.1) -> torch.Tensor:
    """
    Implements a residual connection followed by layer normalization.

    Args:
        x (torch.Tensor): The input tensor.
        sublayer_out (torch.Tensor): The output of the sublayer (e.g., attention or feedforward).
        dropout_p (float): Dropout probability.

    Returns:
        torch.Tensor: The output after the residual connection and layer normalization.
    """
    device = x.device
    d_model = x.shape[-1]
    layernorm = nn.LayerNorm(d_model).to(device)
    dropout = nn.Dropout(dropout_p).to(device)  # create a dropout layer
    out = layernorm(x + dropout(sublayer_out))  # apply dropout to sublayer output
    return out


class MultiHeadedAttention(nn.Module):

    def __init__(self, d_model:int, h:int):
        """
        Implements Multi-Headed Attention.

        Args:
            d_model (int): The dimensionality of the input and output embeddings.
            h (int): The number of attention heads.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by the number of heads"
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.wo = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass of the MultiHeadedAttention layer.

        Args:
            query (torch.Tensor): The query tensor. Shape: (batch_size, seq_len, d_model)
            key (torch.Tensor): The key tensor. Shape: (batch_size, seq_len, d_model)
            value (torch.Tensor): The value tensor. Shape: (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.

        Returns:
            torch.Tensor: The output of the multi-headed attention. Shape: (batch_size, seq_len, d_model)
        """
        batch_size = query.shape[0]
        q = self.wq(query)  # (B,seq_len,d_model)
        k = self.wk(key)  # (B,seq_len,d_model)
        v = self.wv(value)  # (B,seq_len,d_model)

        q = q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        k = k.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        v = v.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)

        output = attention_score(q, k, v, mask)  # (B, h, L, d_k)
        output = output.transpose(1, 2).contiguous()  # (B, L, h, d_k)
        output = output.view(batch_size, -1, self.d_model)  # (B, L, d_model)
        return output


class FeedFoward(nn.Module):

    def __init__(self, d_model:int, d_ff:int):
        """
        Implements a position-wise feedforward network.

        Args:
            d_model (int): The dimensionality of the input and output embeddings.
            d_ff (int): The dimensionality of the hidden layer.
        """
        super(FeedFoward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        """
        Forward pass of the feedforward network.

        Args:
            x (torch.Tensor): The input tensor. Shape: (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: The output tensor. Shape: (batch_size, seq_len, d_model)
        """
        relu_output = self.relu(self.w1(x))
        out = self.w2(relu_output)
        return out


class EncoderLayer(nn.Module):

    def __init__(self, d_model:int, h:int, d_ff):
        """
        Implements a single encoder layer.

        Args:
            d_model (int): The dimensionality of the input and output embeddings.
            h (int): The number of attention heads.
            d_ff (int): The dimensionality of the hidden layer in the feedforward network.
        """
        super(EncoderLayer, self).__init__()

        self.attn = MultiHeadedAttention(d_model, h)
        self.ffn = FeedFoward(d_model, d_ff)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out = sublayer_connection(x, self.attn(x, x, x, mask))
        ffn_out = sublayer_connection(attn_out, self.ffn(attn_out))
        return ffn_out


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, h: int, d_ff: int):
        """
        Implements a single decoder layer.

        Args:
            d_model (int): The dimensionality of the input and output embeddings.
            h (int): The number of attention heads.
            d_ff (int): The dimensionality of the hidden layer in the feedforward network.
        """
        super(DecoderLayer, self).__init__()
        self.masked_attn = MultiHeadedAttention(d_model, h)
        self.cross_attn = MultiHeadedAttention(d_model, h)
        self.ffn = FeedFoward(d_model, d_ff)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, memory_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder layer.

        Args:
            x (torch.Tensor): The input tensor from the previous decoder layer or the target embeddings. 
                                    Shape: (batch_size, seq_len, d_model)
            encoder_out (torch.Tensor): The output tensor from the encoder. Shape: (batch_size, seq_len, d_model)
            memory_mask (torch.Tensor): The padding mask for the encoder output. Shape: (batch_size, 1, 1, seq_len)
            tgt_mask (torch.Tensor): The mask for the target sequence, combining padding and causal masks. 
                                    Shape: (batch_size, 1, seq_len, seq_len)

        Returns:
            torch.Tensor: The output tensor. Shape: (batch_size, seq_len, d_model)
        """
        x = sublayer_connection(x, self.masked_attn(x, x, x, tgt_mask))
        x = sublayer_connection(x, self.cross_attn(x, encoder_out, encoder_out, memory_mask))
        x = sublayer_connection(x, self.ffn(x))
        return x
