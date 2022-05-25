import argparse
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You should experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        ngf = 64
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(100, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Generate images from z
        # z = F.leaky_relu(self.layer[0](z), 0.2)
        # for i in range(1, 4):
        #     layer = self.layer[i]
        #     bnorm = nn.BatchNorm2d(layer.out_features)
        #     z = F.leaky_relu(bnorm(layer(z)), 0.2)
        # z = torch.tanh(self.layer[4](z))
        out = self.layers(z)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You should experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        ngf = 64
        self.layers = nn.Sequential(
            nn.Conv2d(1, ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            nn.Flatten(),
        )

    def forward(self, img):
        device = torch.device('cuda')
        # return discriminator score for img
        # for i in range(2):
        #     img = F.leaky_relu(self.layer[i](img), 0.2)
        # img = torch.tanh(self.layer[2](img))
        out = self.layers(img)
        return out


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    device = torch.device('cuda')
    criterion = nn.BCELoss()
    for epoch in range(200):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            # Train Discriminator
            # ---------------
            optimizer_D.zero_grad()
            z = torch.randn(imgs.size(0), 100).to(device)
            z = z.view(z.size(0), 100, 1, 1)
            f_imgs = generator(z)

            r_label = torch.ones(imgs.size(0), 1).to(device)
            f_label = torch.zeros(imgs.size(0), 1).to(device)

            f_imgs = f_imgs.detach()
            r_output = discriminator(imgs)
            f_output = discriminator(f_imgs)

            f_loss = criterion(f_output, f_label)
            r_loss = criterion(r_output, r_label)

            r_loss.backward()
            f_loss.backward()
            optimizer_D.step()

            # Train Generator
            # -------------------
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), 100).to(device)
            z = z.view(z.size(0), 100, 1, 1)

            f_imgs = generator(z)
            f_output = discriminator(f_imgs)
            gen_loss = criterion(f_output, r_label)
            gen_loss.backward()
            optimizer_G.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % 500 == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                print(f"Epoch {epoch}: loss_g = {gen_loss.item()}, loss_d_real = {r_loss}, loss_d_fake = {f_loss}")
                z = torch.randn(imgs.size(0), 100).to(device)
                z = z.view(z.size(0), 100, 1, 1)
                gen_imgs = generator(z)
                gen_imgs = gen_imgs.view(64, 1, 28, 28)
                a = gen_imgs[:1]
                save_image(gen_imgs[:25],
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)
                pass


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(0.5,
                                                0.5)])),
        batch_size=64, shuffle=True)

    # Initialize models and optimizers
    device = torch.device('cuda')
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
