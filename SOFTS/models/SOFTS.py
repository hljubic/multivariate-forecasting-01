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

class STAR(nn.Module):
    def __init__(self, d_series, d_core, dropout_rate=0.5):
        super(STAR, self).__init__()

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)

        # Adaptive cores
        self.adaptive_core1 = nn.Linear(d_series, d_core // 4)
        self.adaptive_core2 = nn.Linear(d_series, d_core // 4)
        self.adaptive_core3 = nn.Linear(d_series, d_core // 4)
        self.adaptive_core4 = nn.Linear(d_series, d_core // 4)

        # Gating mechanism to control contribution of each core
        self.gate1 = nn.Sigmoid()  # Gate for adaptive_core1
        self.gate2 = nn.Sigmoid()  # Gate for adaptive_core2
        self.gate3 = nn.Sigmoid()  # Gate for adaptive_core3
        self.gate4 = nn.Sigmoid()  # Gate for adaptive_core4
        self.gate_weights = nn.Linear(d_series, 4)  # Generate gating values from input

        # Projection layer to align dimensions before addition
        self.projection = nn.Linear(d_core, d_series)

        self.gen3 = nn.Linear(d_series + d_series, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.activation = nn.ReLU()  # Replace with LASA or custom activation if needed

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        # Apply feed-forward layers
        combined_mean = self.activation(self.gen1(input))
        combined_mean = self.dropout1(combined_mean)  # Apply dropout
        combined_mean = self.gen2(combined_mean)

        # Adaptive Core Formation
        adaptive_core1 = self.adaptive_core1(input.mean(dim=1, keepdim=True))
        adaptive_core2 = self.adaptive_core2(input.mean(dim=1, keepdim=True))
        adaptive_core3 = self.adaptive_core3(input.mean(dim=1, keepdim=True))
        adaptive_core4 = self.adaptive_core4(input.mean(dim=1, keepdim=True))

        # Generate gating values from the input
        gate_values = self.gate_weights(input.mean(dim=1, keepdim=True))  # [B, 1, 4]
        gate1_value, gate2_value, gate3_value, gate4_value = gate_values[:, :, 0], gate_values[:, :, 1], gate_values[:, :, 2], gate_values[:, :, 3]

        # Apply gating mechanisms to control the contribution of each core
        gated_core1 = adaptive_core1 * self.gate1(gate1_value).unsqueeze(2)
        gated_core2 = adaptive_core2 * self.gate2(gate2_value).unsqueeze(2)
        gated_core3 = adaptive_core3 * self.gate3(gate3_value).unsqueeze(2)
        gated_core4 = adaptive_core4 * self.gate4(gate4_value).unsqueeze(2)

        # Combine the gated cores
        combined_gated_cores = gated_core1 + gated_core2 + gated_core3 + gated_core4

        # Project combined gated cores to match the dimension of combined_mean
        combined_gated_cores = self.projection(combined_gated_cores)

        # Add the gated cores to the combined_mean
        combined_mean = combined_mean + combined_gated_cores

        # Concatenate the input and the combined result for the next layer
        combined_mean_cat = torch.cat([input, combined_mean], dim=2)

        combined_mean_cat = self.activation(self.gen3(combined_mean_cat))
        combined_mean_cat = self.dropout2(combined_mean_cat)  # Apply dropout
        combined_mean_cat = self.gen4(combined_mean_cat)

        # Adding residual connection
        output = combined_mean_cat + input

        return output, None

class STAR44(nn.Module):
    def __init__(self, d_series, d_core, dropout_rate=0.5, max_len=5000):
        super(STAR, self).__init__()
        """
        Adaptive STAR with Temporal Embeddings and Dropout
        """

        self.positional_embedding = PositionalEmbedding(d_series, max_len)
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)

        # Adaptive Core Formation
        # Svaka mreža izlazi na d_core // 3
        self.adaptive_core1 = nn.Linear(d_series, d_core // 4)
        self.adaptive_core2 = nn.Linear(d_series, d_core // 4)
        self.adaptive_core3 = nn.Linear(d_series, d_core // 4)
        self.adaptive_core4 = nn.Linear(d_series, d_core // 4)

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
        # Adaptive Core Formation
        adaptive_core1 = self.adaptive_core1(input.mean(dim=1, keepdim=True))
        adaptive_core2 = self.adaptive_core2(input.mean(dim=1, keepdim=True))
        adaptive_core3 = self.adaptive_core3(input.mean(dim=1, keepdim=True))
        adaptive_core4 = self.adaptive_core4(input.mean(dim=1, keepdim=True))

        # Concatenate the four outputs to form a vector of size d_core
        adaptive_core_concat = torch.cat([adaptive_core1, adaptive_core2, adaptive_core3, adaptive_core4], dim=-1)

        # self.activation(
        combined_mean = combined_mean + adaptive_core_concat

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

        # MLP fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = self.activation(self.gen3(combined_mean_cat))
        combined_mean_cat = self.dropout3(combined_mean_cat)  # Apply dropout
        combined_mean_cat = self.gen4(combined_mean_cat)

        # Dodajemo rezidualnu konekciju
        output = combined_mean_cat + input

        return output, None


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
