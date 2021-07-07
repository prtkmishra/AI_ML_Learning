
"""
********************************************************************************************
file: GAN_2D_data.py
author: @Prateek Mishra
Description: GAN implementation for generating 2D analog signal sample
********************************************************************************************
"""
import torch
from torch import nn

import math
import matplotlib.pyplot as plt

torch.manual_seed(111)

train_data_length = 1024
train_data = torch.zeros((train_data_length, 2))
train_data[:, 0] = 4 * math.pi * torch.rand(train_data_length)
train_data[:, 1] = torch.sin(train_data[:, 0])
train_labels = torch.zeros(train_data_length)
train_set = [
    (train_data[i], train_labels[i]) for i in range(train_data_length)
]

plt.plot(train_data[:, 0], train_data[:, 1], ".")

# DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
#            batch_sampler=None, num_workers=0, collate_fn=None,
#            pin_memory=False, drop_last=False, timeout=0,
#            worker_init_fn=None, *, prefetch_factor=2,
#            persistent_workers=False)

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

# NN for Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

discriminator = Discriminator()

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator()

lr = 0.001
num_epochs = 500
loss_function = nn.BCELoss()
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

# Train the GAN

for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples_labels = torch.ones((batch_size, 1)) # torch.ones() to create labels with the value 1 for the real samples
        latent_space_samples = torch.randn((batch_size, 2)) # Noise for generator samples
        generated_samples = generator(latent_space_samples) # Generator samples
        generated_samples_labels = torch.zeros((batch_size, 1)) # Label the generator samples as zero
        all_samples = torch.cat((real_samples, generated_samples)) # Concatenate all the samples
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        ) # Concatenate sample labels

        # Training the discriminator
        discriminator.zero_grad() #In PyTorch, it’s necessary to clear the gradients at each training step to avoid accumulating them
        output_discriminator = discriminator(all_samples) # Calculate the output of the discriminator using the training data in all_samples.
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        
        loss_discriminator.backward() # Calculate the gradients by backward()
        optimizer_discriminator.step() # update the discriminator weights

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 2))

        # Training the generator
        generator.zero_grad() # clear the gradients 
        generated_samples = generator(latent_space_samples) #Input only the generator samples
        
        output_discriminator_generated = discriminator(generated_samples) # Feed the generator’s output into the discriminator and store its output in output_discriminator_generated
        
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        ) # Calculate loss for generator
        
        loss_generator.backward() # Calculate the gradients by backward()
        optimizer_generator.step() # update the discriminator weights
        latent_space_samples = torch.randn(100, 2)
        # Show loss
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")