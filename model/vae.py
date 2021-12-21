import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, in_channels=1, z_dim=64):
        super(VAE, self).__init__()
        hidden_dims = [in_channels] + [4, 16]

        self.encoder = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=hidden_dims[i],
                        out_channels=hidden_dims[i + 1],
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.fc_mu = nn.Linear(hidden_dims[-1] * 7 * 7, z_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 7 * 7, z_dim)
        self.decoder_input = nn.Linear(z_dim, hidden_dims[-1] * 7 * 7)

        self.decoder = nn.ModuleList()
        for i in range(len(hidden_dims) - 2):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_dims[-i - 1],
                        out_channels=hidden_dims[-i - 2],
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=(1, 1),
                        output_padding=(1, 1),
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[1],
                hidden_dims[1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1),
            ),
            nn.BatchNorm2d(hidden_dims[-2]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_dims[1],
                      out_channels=hidden_dims[0],
                      kernel_size=(3, 3),
                      padding=(1, 1)
                      ),
            nn.Tanh(),
        )

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def decode(self, z):
        y = self.decoder_input(z)
        y = y.view(-1, 16, 7, 7)

        for layer in self.decoder:
            y = layer(y)
        y = self.final_layer(y)
        return y

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y = self.decode(z)
        return y, x, mu, log_var
