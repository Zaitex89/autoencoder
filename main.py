import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader # type: ignore
from torchvision import datasets, transforms # type: ignore
import matplotlib.pyplot as plt

# 1. Load the data

# MNIST images are 28x28 grayscale = 784 pixels when flattened into a vector.
# ToTensor() scales pixel values into the [0, 1] range.
transform = transforms.ToTensor()

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

# 2. Define the autoencoder

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # ENCODER: squeezes 784 pixels down to "latent_dim" numbers
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim), # the bottleneck
        )

        # DECODER: expands the latent code back up to 784 pixels from latent dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(), # keeps outputs in [0, 1] like the input
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


# 3. Train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(latent_dim=32).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 20
for epoch in range(EPOCHS):
    running_loss = 0.0
    for images, _ in train_loader: # note we ignore the labels (_)
        images = images.view(images.size(0), -1).to(device) # flatten to 784

        reconstructed = model(images)
        loss = criterion(reconstructed, images) # compare OUTPUT to the INPUT

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} loss: {avg:.4f}")

# 4. See how well it reconstructs digits it never trained on

model.eval()
images, _ = next(iter(test_loader))
flat = images.view(images.size(0), -1).to(device)

with torch.no_grad():
    outputs = model(flat).cpu().view(-1, 28, 28)

# Top row = originals, bottom row = reconstructions.
n = 8
fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
for i in range(n):
    axes[0][i].imshow(images[i][0], cmap="gray")
    axes[0][i].axis("off")
    axes[1][i].imshow(outputs[i], cmap="gray")
    axes[1][i].axis("off")
axes[0][0].set_ylabel("Original")
axes[1][0].set_ylabel("Rebuilt")
plt.tight_layout()
plt.savefig("reconstructions.png", dpi=120)
print("Saved reconstructions.png")
plt.show()