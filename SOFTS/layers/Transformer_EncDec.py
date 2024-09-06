import torch.nn as nn
import torch.nn.functional as F
import torch
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


class LACU2(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(LACU, self).__init__()
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

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", **kwargs):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()#LACU() #F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta,
            **kwargs
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, **kwargs)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
