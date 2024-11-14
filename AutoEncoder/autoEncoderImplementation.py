import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
class Autoencoder(nn.Module):
   def __init__(self, encoding_dim):
       super(Autoencoder, self).__init__()
       self.encoder = nn.Sequential(
           nn.Conv2d(1, 32, kernel_size=5),
           nn.ReLU(True),
           nn.Conv2d(32,encoding_dim,kernel_size=5),
           nn.ReLU(True))
       self.decoder = nn.Sequential(
           nn.ConvTranspose2d(encoding_dim,32,kernel_size=5),
           nn.ReLU(True),
           nn.ConvTranspose2d(32,1,kernel_size=5),
           nn.ReLU(True),
           nn.Sigmoid())
   def forward(self,x):
       x = self.encoder(x)
       x = self.decoder(x)
       return x

# Convert data to torch.FloatTensor
transform = transforms.ToTensor()

# Load the training and test set
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Create training and test dataloaders
# Number of subprocesses to use for data loading
num_workers = 0
# How many samples per batch to load
batch_size = 42

# Prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# Obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy()

# Get one image from the batch
img = np.squeeze(images[0])

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')

encoding_dim = 64
model = Autoencoder(encoding_dim)
print(model)

# specify loss function
criterion = nn.MSELoss()

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# number of epochs to train the model
n_epochs = 1

for epoch in range(1, n_epochs+1):
   # monitor training loss
   train_loss = 0.0
   for data in train_loader:
       # _ stands in for labels, here
       images, _ = data
       # clear the gradients of all optimized variables
       optimizer.zero_grad()
       # forward pass: compute predicted outputs by passing inputs to the model
       outputs = model(images)
       # calculate the loss
       loss = criterion(outputs, images)

       # backward pass: compute gradient of the loss with respect to model parameters
       loss.backward()
       # perform a single optimization step (parameter update)
       optimizer.step()
       # update running training loss
       train_loss += loss.item() * images.size(0)

       # print avg training statistics
       train_loss = train_loss / len(train_loader)
       print('Epoch: {} \tTraining Loss: {:.6f}'.format(
           epoch,
           train_loss
       ))

# Obtain one batch of test images
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Get sample outputs
output = model(images)
output = output.view(batch_size, 1, 28, 28)
output = output.detach().numpy()

# Plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
for images, row in zip([images.numpy(), output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

# Wait for a key press and close the display windows
#cv2.waitKey(0)
#cv2.destroyAllWindows()

