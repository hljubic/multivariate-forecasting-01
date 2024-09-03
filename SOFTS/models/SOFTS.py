import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer


class LearnableAsymCauchy44(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(LearnableAsymCauchy44, self).__init__()
        # Inicijalizacija parametara kao trenirajući parametri
        self.alpha = 1.3#nn.Parameter(torch.tensor(alpha))
        self.beta = 0.7#nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        alpha = 1.3#nn.Parameter(torch.tensor(alpha))
        beta = 0.7#nn.Parameter(torch.tensor(beta))
        pos_part = 1 / (1 + alpha * torch.relu(x) ** 2)
        neg_part = 1 / (1 + beta * torch.relu(-x) ** 2)
        return torch.exp(pos_part - neg_part)

class LeakyCustomActivation(nn.Module):
    def __init__(self, negative_slope=0.2, positive_slope=0.2):
        super(LeakyCustomActivation, self).__init__()
        self.negative_slope = negative_slope
        self.positive_slope = positive_slope

    def forward(self, x):
        # Apply the leaky custom piecewise function
        out = torch.where(x <= -1, self.negative_slope * (x + 1),
                          torch.where(x >= 1, self.positive_slope * (x - 1) + 1, 0.5 * x + 0.5))
        return out
class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        # Apply the custom piecewise function
        out = torch.where(x <= -1, torch.zeros_like(x),
                          torch.where(x >= 1, torch.ones_like(x), 0.5 * x + 0.5))
        return out

class LearnableAsymCauchy(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(LearnableAsymCauchy, self).__init__()
        # Inicijalizacija parametara kao trenirajući parametri
        self.alpha = 1.3#nn.Parameter(torch.tensor(alpha))
        self.beta = 0.7#nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        pos_part = 1 / (1 + self.alpha * torch.relu(x) ** 2)
        neg_part = 1 / (1 + self.beta * torch.relu(-x) ** 2)
        return pos_part - neg_part

class AsymCauchy(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(AsymCauchy, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        # Apply the piecewise AsymCauchy function
        pos_part = 1 / (1 + self.alpha * x.pow(2))
        neg_part = -1 / (1 + self.beta * x.pow(2))
        return torch.where(x >= 0, pos_part, neg_part)

class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)
        
        self.activation = nn.Sigmoid()

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        # set FFN
        combined_mean = self.activation(self.gen1(input))
        combined_mean = self.gen2(combined_mean)

        # stochastic pooling
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

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = self.activation(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        output = combined_mean_cat

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
