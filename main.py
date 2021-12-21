import os
import warnings

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from data.dataset import Dataset
from model.loss import Loss
from model.vae import VAE
from utils.setseed import set_seed

warnings.filterwarnings("ignore")

lr = 0.0001
epochs = 30
batch_size = 64
z_dim = 64
rootpath = "./"
imgpath = rootpath + 'img'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    set_seed(0)

    dataset = Dataset()

    trainset = dataset.trainset
    testset = dataset.testset

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = VAE(z_dim=z_dim).to(device)
    criterion = Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # train
    for e in range(epochs):
        model.train()
        trainlosses = []
        for (x, _) in tqdm(trainloader):
            x = x.to(device)
            out, input, mu, log_var = model(x)
            loss = criterion(out, input, mu, log_var)
            trainlosses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            testlosses = []
            for (x, _) in testloader:
                x = x.to(device)
                out, input, mu, log_var = model(x)
                loss = criterion(out, input, mu, log_var)
                testlosses.append(loss.item())

        print("Epochs:", e, " || train loss: %.4f" % np.mean(trainlosses), " || test loss: %.4f" % np.mean(testlosses))

        model.eval()
        with torch.no_grad():
            for (x, _) in testloader:
                x = x.to(device)
                out, input, mu, log_var = model(x)
                concat = torch.cat([input.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
                save_image(concat, os.path.join(imgpath, 'reconst-epoch-{}.png'.format(e + 1)))
                break

    torch.save(model, rootpath + "log/vae.pkl")

    # sample
    model.eval()
    with torch.no_grad():
        z = torch.randn(batch_size, z_dim).to(device)
        out = model.decode(z)
        save_image(out, os.path.join(imgpath, 'sample.png'))
