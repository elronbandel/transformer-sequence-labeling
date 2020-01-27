import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SequenceTaggingTransformer(nn.Module):

    def __init__(self, vocab_size, num_tags, embed_dim, num_heads, hid_dim, num_layers, dropout=0.5):
        super(SequenceTaggingTransformer, self).__init__()
        self.padder = SequencePadding(padding_value=vocab_size)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, num_heads, hid_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.embedder = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size)
        self.embed_dim = embed_dim
        self.decoder = nn.Linear(embed_dim, num_tags + 1)

        self.init_weights()
    def normalizer(self, input):
        return input * math.sqrt(self.embed_dim)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, sequences):
        padded, lens = self.padder(sequences)
        embeded = self.normalizer(self.embedder(padded))
        encoded = self.pos_encoder(embeded)
        transformed = self.transformer_encoder(encoded)
        output = self.decoder(transformed)
        return output


class SequencePadding(nn.Module):
    def __init__(self, padding_value=0, batch_first=False, device=None):
        super().__init__()
        self.batch_first = batch_first
        self.padding_value = padding_value
        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, sequences):
        lens = torch.LongTensor(list(map(len, sequences)))
        padded = pad_sequence(sequences, batch_first=self.batch_first, padding_value=self.padding_value).to(self.device)
        return padded, lens
