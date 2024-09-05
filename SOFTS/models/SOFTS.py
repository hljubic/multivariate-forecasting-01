import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer


class LASA(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(LASA, self).__init__()
        # Inicijalizacija parametara kao trenirajući parametri
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        alpha = self.alpha
        beta = self.beta # Linearni prijelaz za vrijednosti blizu nule

        # Izbjegavamo višestruke pozive relu funkciji i kombinujemo operacije
        relu_x = torch.relu(x)
        relu_neg_x = torch.relu(-x)

        # Direktna primjena u formuli
        pos_part = 1 / (1 + alpha * relu_neg_x ** 2)
        neg_part = 1 / (1 + beta * relu_x ** 2)

        return pos_part - neg_part

class PositionalEmbedding(nn.Module):
    def __init__(self, d_series, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, d_series), requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_series, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_series))
        self.position_embedding[:, :, 0::2] = torch.sin(position * div_term)
        self.position_embedding[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.position_embedding[:, :x.size(1)]
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class STAR(nn.Module):
    def __init__(self, d_series, d_core, dropout_rate=0.5, max_len=5000, sigma=1.0):
        super(STAR, self).__init__()
        """
        Adaptive STAR with Temporal Embeddings, Dropout, and Gaussian Smoothing for differences.
        """

        # Positional embedding for trend and differences
        self.positional_embedding = PositionalEmbedding(d_series, max_len)

        # Trend branch
        self.trend_gen1 = nn.Linear(d_series, d_series)
        self.trend_gen2 = nn.Linear(d_series, d_core)
        self.adaptive_core_trend = nn.Linear(d_series, d_core)

        # Difference branch
        self.diff_gen1 = nn.Linear(d_series, d_series)
        self.diff_gen2 = nn.Linear(d_series, d_core)
        self.adaptive_core_diff = nn.Linear(d_series, d_core)

        # Gaussian filter for smoothing diff_out
        self.gaussian_filter = self.create_gaussian_filter(sigma, kernel_size=5)

        # Fusion layers
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.activation = LASA()  # Assuming LASA is a custom activation function

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        # Apply positional embedding
        input = self.positional_embedding(input)

        # === Trend Branch ===
        trend_out = self.activation(self.trend_gen1(input))
        trend_out = self.dropout1(trend_out)
        trend_out = self.trend_gen2(trend_out)

        # Adaptive Core for trend
        adaptive_core_trend = self.adaptive_core_trend(input.mean(dim=1, keepdim=True))
        trend_out = trend_out + adaptive_core_trend

        # === Difference Branch ===
        diff_out = self.activation(self.diff_gen1(input))
        diff_out = self.dropout1(diff_out)
        diff_out = self.diff_gen2(diff_out)

        # Adaptive Core for differences
        adaptive_core_diff = self.adaptive_core_diff(input.mean(dim=1, keepdim=True))
        diff_out = diff_out + adaptive_core_diff

        # Apply Gaussian Smoothing to diff_out
        diff_out = diff_out.permute(0, 2, 1)  # Change shape for Conv1d (batch_size, channels, sequence_length)
        diff_out = self.gaussian_filter(diff_out)
        diff_out = diff_out.permute(0, 2, 1)  # Return to original shape

        # Stochastic pooling for trend and diff (if training)
        if self.training:
            trend_out = self._stochastic_pooling(trend_out, batch_size, channels)
            diff_out = self._stochastic_pooling(diff_out, batch_size, channels)
        else:
            trend_out = self._weighted_sum(trend_out, channels)
            diff_out = self._weighted_sum(diff_out, channels)

        # Combine trend and differences
        combined_mean = trend_out + diff_out

        combined_mean = self.dropout2(combined_mean)

        # MLP fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = self.activation(self.gen3(combined_mean_cat))
        combined_mean_cat = self.dropout3(combined_mean_cat)
        combined_mean_cat = self.gen4(combined_mean_cat)

        # Add residual connection
        output = combined_mean_cat + input

        return output, None

    def _stochastic_pooling(self, combined_mean, batch_size, channels):
        ratio = F.softmax(combined_mean, dim=1)
        ratio = ratio.permute(0, 2, 1)
        ratio = ratio.reshape(-1, channels)
        indices = torch.multinomial(ratio, 1)
        indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
        pooled_mean = torch.gather(combined_mean, 1, indices)
        return pooled_mean.repeat(1, channels, 1)

    def _weighted_sum(self, combined_mean, channels):
        weight = F.softmax(combined_mean, dim=1)
        weighted_sum = torch.sum(combined_mean * weight, dim=1, keepdim=True)
        return weighted_sum.repeat(1, channels, 1)

    def create_gaussian_filter(self, sigma, kernel_size):
        # Create a Gaussian kernel
        kernel = torch.tensor([math.exp(-(x - kernel_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(kernel_size)])
        kernel = kernel / kernel.sum()  # Normalize kernel

        # Reshape to 1D convolutional weight
        kernel = kernel.view(1, 1, -1)  # Shape: (out_channels, in_channels, kernel_size)
        gaussian_filter = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        gaussian_filter.weight.data = kernel
        gaussian_filter.weight.requires_grad = False  # Kernel is not trainable
        return gaussian_filter


class STAR2(nn.Module):
    def __init__(self, d_series, d_core, dropout_rate=0.5, max_len=5000):
        super(STAR, self).__init__()
        """
        Adaptive STAR with Temporal Embeddings and Dropout
        """

        self.positional_embedding = PositionalEmbedding(d_series, max_len)
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)

        # Adaptive Core Formation
        self.adaptive_core = nn.Linear(d_series, d_core)

        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.activation = LASA()

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        # Apply temporal embedding
        input = self.positional_embedding(input)

        # Set FFN
        combined_mean = self.activation(self.gen1(input))
        combined_mean = self.dropout1(combined_mean)  # Apply dropout
        combined_mean = self.gen2(combined_mean)

        # Adaptive Core Formation
        adaptive_core = self.adaptive_core(input.mean(dim=1, keepdim=True))
        combined_mean = combined_mean + adaptive_core

        # Stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        combined_mean = self.dropout2(combined_mean)  # Apply dropout

        # mlp fusion
        # Rezidualna konekcija s ulaznim podacima
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = self.activation(self.gen3(combined_mean_cat))
        combined_mean_cat = self.dropout3(combined_mean_cat)  # Apply dropout
        combined_mean_cat = self.gen4(combined_mean_cat)

        # Dodajemo rezidualnu konekciju
        output = combined_mean_cat + input

        return output, None


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.use_norm = configs.use_norm
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    STAR(configs.d_model, configs.d_core),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
        )

        # Decoder
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
