import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer

class LACU(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(LACU, self).__init__()
        # Inicijalizacija parametara kao trenirajući parametri
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        # Kombinovanje operacija i eliminacija suvišnih relu poziva
        relu_x = torch.clamp(x, min=0)  # Zamena za torch.relu(x)
        relu_neg_x = torch.clamp(-x, min=0)  # Zamena za torch.relu(-x)

        # Direktna primena kvadriranja
        relu_neg_x_sq = relu_neg_x * relu_neg_x
        relu_x_sq = relu_x * relu_x

        # Kombinovanje računa
        pos_part = 1 / (1 + self.alpha * relu_neg_x_sq)
        neg_part = 1 / (1 + self.beta * relu_x_sq)

        # Konačni rezultat
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


class CloudBlock(nn.Module):
    def __init__(self, d_series, d_core, num_layers=2, dropout_rate=0.5):
        super(CloudBlock, self).__init__()
        """
        Cloud block that processes data before the rain layers
        """
        # Oblak je sastavljen od nekoliko linearnih slojeva koji transformišu podatke
        self.layers = nn.ModuleList([nn.Linear(d_series, d_core) for _ in range(num_layers)])
        self.activation = LACA()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = self.activation(layer(out))
            out = self.dropout(out)
        return out

class STAR(nn.Module):
    def __init__(self, d_series, d_core, num_layers=5, num_cloud_layers=3, dropout_rate=0.5, max_len=5000):
        super(STAR, self).__init__()
        """
        Rain-like Network Architecture with Temporal Embeddings and Dropout
        """

        # Pozicijska embeding komponenta
        self.positional_embedding = PositionalEmbedding(d_series, max_len)

        # Oblaci (clouds) preprocesiraju podatke pre kišnih slojeva
        self.cloud = CloudBlock(d_series, d_core, num_layers=num_cloud_layers, dropout_rate=dropout_rate)

        # Slojevi (nivoa kiše), svaki sloj predstavlja "padanje" kapljica ka dole
        self.layers = nn.ModuleList([nn.Linear(d_core, d_core) for _ in range(num_layers)])

        # Poslednji sloj koji predstavlja tlo (spajanje svih kapljica u jednu)
        self.output_layer = nn.Linear(d_core, d_series)

        # Dropout slojevi
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers)])

        # Aktivacija
        self.activation = LACA()

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        # Primjena temporalnog embeddinga, što može predstavljati "start" kišnih kapljica
        input = self.positional_embedding(input)

        # Procesiranje podataka kroz oblake (clouds)
        cloud_out = self.cloud(input)

        # Kreiramo akumulator za zbrajanje izlaza iz svakog sloja
        cumulative_output = torch.zeros_like(cloud_out)

        # Padanje kroz slojeve, svaki sloj je kao nivo gde kiša pada na različite dijelove podataka
        out = cloud_out
        for layer, dropout in zip(self.layers, self.dropouts):
            out = self.activation(layer(out))
            out = dropout(out)

            # Akumulacija izlaza iz svakog sloja
            cumulative_output += out

        # Konačno, svi podaci stižu do izlaznog sloja ("tlo")
        out = self.output_layer(cumulative_output)

        # Dodajemo rezidualnu konekciju koja spaja početni ulaz s krajnjim izlazom
        output = out + input

        return output, None


class STA2R(nn.Module):
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

        self.activation = LACU()

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        # Apply temporal embedding
        input = self.positional_embedding(input)

        return self.gen4(self.dropout1(self.activation(self.gen4(input)))) + input, None

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
