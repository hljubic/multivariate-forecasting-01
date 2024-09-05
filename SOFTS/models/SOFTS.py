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

class TemporalEmbedding(nn.Module):
    def __init__(self, d_series, max_len=5000):
        super(TemporalEmbedding, self).__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, d_series), requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_series, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_series))
        self.position_embedding[:, :, 0::2] = torch.sin(position * div_term)
        self.position_embedding[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.position_embedding[:, :x.size(1)]
        return x

class STAR(nn.Module):
    def __init__(self, d_series, d_core, dropout_rate=0.5, max_len=5000):
        super(STAR, self).__init__()
        """
        Adaptive STAR with Temporal Embeddings and Dropout
        """

        self.temporal_embedding = TemporalEmbedding(d_series, max_len)
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
        input = self.temporal_embedding(input)

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

class STAR2(nn.Module):
    def __init__(self, d_series, d_core, dropout_rate=0.1):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

        # Dropout layers
        self.dropout = nn.Dropout(p=dropout_rate)

        self.activation = self.activation#LASA()

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        # set FFN
        combined_mean = self.activation(self.gen1(input))
        combined_mean = self.dropout(combined_mean)  # Apply dropout
        combined_mean = self.gen2(combined_mean)

        # stochastic pooling
        if self.training:
            #ratio = F.softmax(combined_mean, dim=1)
            #ratio = ratio.permute(0, 2, 1)
            # ratio = ratio.reshape(-1, channels)
            ratio = F.softmax(combined_mean, dim=1).permute(0, 2, 1).reshape(-1, channels)

            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        # mlp fusion
        # Rezidualna konekcija s ulaznim podacima
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = self.activation(self.gen3(combined_mean_cat))
        combined_mean_cat = self.dropout(combined_mean_cat)  # Apply dropout
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

    def gaussian_filter(self, input_tensor, kernel_size, sigma):
        """
        Apply a Gaussian filter to smooth the input_tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor to be smoothed (batch_size, seq_len, num_features).
            kernel_size (int): The size of the Gaussian kernel.
            sigma (float): The standard deviation of the Gaussian distribution.

        Returns:
            torch.Tensor: The smoothed tensor.
        """
        # Create a 1D Gaussian kernel
        kernel = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
        kernel = kernel / kernel.sum()  # Normalize the kernel to ensure sum is 1

        # Ensure the kernel is expanded for each feature (group)
        kernel = kernel.view(1, 1, -1).to(input_tensor.device)

        # Apply Gaussian filter for each feature across the sequence dimension
        num_features = input_tensor.size(2)
        smoothed_tensor = []
        for i in range(num_features):
            # Select the i-th feature (sequence length along the dimension 1)
            feature_tensor = input_tensor[:, :, i].unsqueeze(1)  # [batch_size, 1, seq_len]
            smoothed_feature = F.conv1d(feature_tensor, kernel, padding=(kernel_size - 1) // 2)
            smoothed_tensor.append(smoothed_feature)

        # Concatenate along the feature dimension (dim=2)
        smoothed_tensor = torch.cat(smoothed_tensor, dim=1).permute(0, 2, 1)

        return smoothed_tensor

    def estimate_frequency(self, data):
        # Estimate frequency using FFT (Fast Fourier Transform)
        fft_result = torch.fft.rfft(data, dim=1)
        # Take the magnitude of the frequencies
        frequencies = torch.abs(fft_result)
        # Calculate average frequency magnitude
        avg_frequency = torch.mean(frequencies, dim=1)
        return avg_frequency

    def normalize_frequencies(self, data, target_frequency):
        # Estimate the frequency of each sequence
        frequencies = self.estimate_frequency(data)

        # Calculate scaling factors based on how far each sequence is from the target frequency
        scaling_factors = frequencies / target_frequency

        # Apply Gaussian filter with inverse scaling factor (stronger smoothing for higher frequency sequences)
        length = data.size(1)
        for i in range(data.size(0)):
            sigma = 1.0 / scaling_factors[i].item() if scaling_factors[i].numel() == 1 else 1.0 / scaling_factors[i][
                0].item()  # Handle the case of multi-element tensors
            data[i, :, :] = self.gaussian_filter(data[i:i + 1, :, :], kernel_size=9, sigma=sigma)

        return data
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x_enc = self.normalize_frequencies(x_enc, target_frequency=1.5)

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
