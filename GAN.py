import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import imageio
from PIL import Image
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,)),
                transforms.CenterCrop((652,502)),
                ])
path="H:/GAN-Kumar/box"
dataset = ImageFolder(path, transform=transform)
dataloader = DataLoader(dataset, batch_size=37, shuffle=True)
to_image = transforms.ToPILImage()
trainloader = DataLoader(dataset, batch_size=100, shuffle=True)

device = 'cpu'
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.n_features = 64
        self.n_out = 652*502
        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_features, 128),
                    nn.LeakyReLU(0.2)
                    )
        self.fc1 = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.LeakyReLU(0.2)
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.LeakyReLU(0.2)
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(512, self.n_out),
                    nn.Tanh()
                    )
    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view(-1, 1,652, 502)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_in = 652*502
        self.n_out = 1
        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_in, 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1)
                    )
        self.fc1 = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1)
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1)
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(128, self.n_out),
                    nn.Sigmoid()
                    )
    def forward(self, x):
        x = x.view(-1, 652*502)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
generator = Generator()
discriminator = Discriminator()
generator.to(device)
discriminator.to(device)

g_optim = optim.Adam(generator.parameters(), lr=2e-4)
d_optim = optim.Adam(discriminator.parameters(), lr=2e-4)

g_losses = []
d_losses = []
images = []

criterion = nn.BCELoss()

def noise(n, n_features=64):
    return Variable(torch.randn(n, n_features)).to(device)

def make_ones(size):
    data = Variable(torch.ones(size, 1))
    return data.to(device)

def make_zeros(size):
    data = Variable(torch.zeros(size, 1))
    return data.to(device)
def train_discriminator(optimizer, real_data, fake_data):
    n = real_data.size(0)
    optimizer.zero_grad()
    
    prediction_real = discriminator(real_data)
    error_real = criterion(prediction_real, make_ones(n))
    error_real.backward()

    prediction_fake = discriminator(fake_data)
    error_fake = criterion(prediction_fake, make_zeros(n))
    
    error_fake.backward()
    optimizer.step()
    
    return error_real + error_fake

def train_generator(optimizer, fake_data):
    n = fake_data.size(0)
    optimizer.zero_grad()
    
    prediction = discriminator(fake_data)
    error = criterion(prediction, make_ones(n))
    
    error.backward()
    optimizer.step()
    
    return error
num_epochs = 500
k = 1
test_noise = noise(1)
image = None 
g_losses=[]
d_losses=[]
generator.train()
discriminator.train()
for epoch in range(num_epochs):
    g_error = 0.0
    d_error = 0.0
    for i, data in enumerate(trainloader):
        imgs, _ = data
        n = len(imgs)
        for j in range(k):
            fake_data = generator(noise(n)).detach()
            real_data = imgs.to(device)
            d_error += train_discriminator(d_optim, real_data, fake_data)
        fake_data = generator(noise(n))
        g_error += train_generator(g_optim, fake_data)
    if epoch == 499:
        img = generator(test_noise).cpu().detach()
        img = make_grid(img)
    
    print('Epoch {}: g_loss: {:.8f} d_loss: {:.8f}\r'.format(epoch, g_error, d_error))
    
print('Training Finished')
#torch.save(generator.state_dict(), 'mnist_generator.pth')

import numpy as np
from matplotlib import pyplot as plt
imgs = np.array(to_image(img))
img=Image.fromarray(imgs)
img.save("H:\GAN-Kumar\img.jpg")
