import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim   # för SSIM


# ==================== Samma modell som i main.py ====================
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)


# ==================== Utvärdering ====================
def evaluate_reconstructions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Använder enhet: {device}\n")

    # Data
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    # Modell (samma som i main.py)
    model = Autoencoder(latent_dim=32).to(device)
    
    # OBS: Eftersom main.py inte sparar vikterna tränar vi om snabbt här
    # (eller kör main.py först och spara modell om du vill)
    print("Tränar modell snabbt för utvärdering (5 epochs)...")
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(5):  # kort träning bara för att kunna utvärdera
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("Klar med snabbträning.\n")

    # ===== Utvärdering =====
    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    ssim_total = 0.0
    n_samples = 0

    all_originals = []
    all_recons = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images_flat = images.view(images.size(0), -1).to(device)
            recons_flat = model(images_flat)
            
            # MSE & MAE
            mse = torch.mean((recons_flat - images_flat) ** 2, dim=1)
            mae = torch.mean(torch.abs(recons_flat - images_flat), dim=1)
            
            mse_total += mse.sum().item()
            mae_total += mae.sum().item()
            n_samples += images.size(0)

            # SSIM (på bildnivå)
            recons_img = recons_flat.cpu().view(-1, 28, 28).numpy()
            orig_img = images.squeeze().numpy()
            
            for i in range(len(recons_img)):
                ssim_val = ssim(orig_img[i], recons_img[i], data_range=1.0)
                ssim_total += ssim_val

            all_originals.append(images)
            all_recons.append(recons_flat.cpu().view(-1, 1, 28, 28))
            all_labels.append(labels)

    # Genomsnittliga metrics
    avg_mse = mse_total / n_samples
    avg_mae = mae_total / n_samples
    avg_ssim = ssim_total / n_samples

    print("=== Utvärdering: Rekonstruerade vs Original ===")
    print(f"Antal testbilder:     {n_samples}")
    print(f"Genomsnittlig MSE:    {avg_mse:.6f}")
    print(f"Genomsnittlig MAE:    {avg_mae:.6f}")
    print(f"Genomsnittlig SSIM:   {avg_ssim:.4f}  (1.0 = perfekt)")

    # ===== Visualisering med skillnad =====
    originals = torch.cat(all_originals)[:10]
    recons = torch.cat(all_recons)[:10]
    labels = torch.cat(all_labels)[:10]

    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    for i in range(10):
        # Original
        axes[0, i].imshow(originals[i].squeeze(), cmap="gray")
        axes[0, i].set_title(f"Orig: {labels[i].item()}")
        axes[0, i].axis("off")

        # Reconstructed
        axes[1, i].imshow(recons[i].squeeze(), cmap="gray")
        axes[1, i].set_title("Recon")
        axes[1, i].axis("off")

        # Skillnad (absolut error)
        diff = torch.abs(originals[i] - recons[i])
        axes[2, i].imshow(diff.squeeze(), cmap="hot")
        axes[2, i].set_title("Skillnad")
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    axes[2, 0].set_ylabel("Absolut fel", fontsize=12)

    plt.suptitle(f"Jämförelse (MSE: {avg_mse:.5f} | SSIM: {avg_ssim:.3f})")
    plt.tight_layout()
    plt.savefig("evaluation_comparison.png", dpi=150)
    plt.show()
    print("\nJämförelsebild sparad som: evaluation_comparison.png")


if __name__ == "__main__":
    evaluate_reconstructions()