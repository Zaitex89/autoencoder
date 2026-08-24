import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim   # for SSIM


# same model as main.py
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


# Evaluation
def evaluate_reconstructions(model_path="autoencoder.pth", latent_dim=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Data (only test data is needed now – no training here)
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    # Build the model and load in the saved weights from main.py.
    # NOTE: latent_dim needs to be the same here as when the model was trained,
    # otherwise the weights don't match the layers.
    model = Autoencoder(latent_dim=latent_dim).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot find '{model_path}'. Run main.py first so the model is saved."
        )

    # map_location makes it work regardless of whether the model was saved on GPU or CPU
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}\n")

    # evaluation
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

            # SSIM (per image)
            recons_img = recons_flat.cpu().view(-1, 28, 28).numpy()
            orig_img = images.squeeze().numpy()

            for i in range(len(recons_img)):
                ssim_val = ssim(orig_img[i], recons_img[i], data_range=1.0)
                ssim_total += ssim_val

            all_originals.append(images)
            all_recons.append(recons_flat.cpu().view(-1, 1, 28, 28))
            all_labels.append(labels)

    # Average metrics
    avg_mse = mse_total / n_samples
    avg_mae = mae_total / n_samples
    avg_ssim = ssim_total / n_samples

    print("=== Evaluation: Reconstructed vs Original ===")
    print(f"Number of test images: {n_samples}")
    print(f"Average MSE:           {avg_mse:.6f}")
    print(f"Average MAE:           {avg_mae:.6f}")
    print(f"Average SSIM:          {avg_ssim:.4f}  (1.0 = perfect)")

    # Visualization with a difference row
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

        # Difference (absolute error)
        diff = torch.abs(originals[i] - recons[i])
        axes[2, i].imshow(diff.squeeze(), cmap="hot")
        axes[2, i].set_title("Difference")
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    axes[2, 0].set_ylabel("Absolute error", fontsize=12)

    plt.suptitle(f"Comparison (MSE: {avg_mse:.5f} | SSIM: {avg_ssim:.3f})")
    plt.tight_layout()
    plt.savefig("evaluation_comparison.png", dpi=150)
    plt.show()
    print("\nComparison image saved as: evaluation_comparison.png")


if __name__ == "__main__":
    evaluate_reconstructions()