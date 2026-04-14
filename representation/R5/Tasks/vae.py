import numpy as np
import torch
from torch import nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.config = {}

    @classmethod
    def load(cls, path, map_location="cpu", restore_weights=True):
        data = torch.load(path, map_location=map_location)
        model = cls(**data["config"])
        if restore_weights:
            model.load_state_dict(data["weights"])
        return model

    def to(self, device):
        self.device = device
        super().to(device)
        return self


class VAE_PianoFingering(BaseModel):
    def __init__(self, embedding_size, layer_sizes, latent_size):
        super().__init__()
        self.config.update({
            "embedding_size": embedding_size,
            "layer_sizes": layer_sizes,
            "latent_size": latent_size
        })
        self.embedding_size = embedding_size
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            *self.DNN(self.embedding_size, *layer_sizes, 2 * latent_size)
        )

        self.decoder = nn.Sequential(
            *self.DNN(latent_size, *layer_sizes[::-1], embedding_size),
            nn.Sigmoid(),
        )

    @staticmethod
    def DNN(*layers):
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            if i != len(layers) - 2:
                net.append(nn.LeakyReLU())
        return net

    def sample(self, batch_size = 1, z=None):
        if z is None:
            z = torch.randn(batch_size, self.latent_size)
        z = z.to(self.device)
        return self.decoder(z)