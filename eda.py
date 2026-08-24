import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import Counter


class MNISTEDA:
    def __init__(self):
        transform = transforms.ToTensor()
        self.train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        self.test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    def summary(self):
        print("=== EDA: MNIST Dataset ===")
        print(f"Train samples: {len(self.train_data)}")
        print(f"Test samples:  {len(self.test_data)}")

        train_labels = [label for _, label in self.train_data]
        print("\nKlassfördelning (train):")
        print(dict(Counter(train_labels)))

        test_labels = [label for _, label in self.test_data]
        print("\nKlassfördelning (test):")
        print(dict(Counter(test_labels)))

    def show_examples(self, save_path="eda_examples.png"):
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i, ax in enumerate(axes.flat):
            img, label = self.train_data[i]
            ax.imshow(img.squeeze(), cmap="gray")
            ax.set_title(f"Label: {label}")
            ax.axis("off")
        plt.suptitle("Exempelbilder från MNIST")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.show()
        print(f"Exempelbilder sparade som: {save_path}")

    def show_class_averages(self, save_path="eda_class_averages.png"):
        class_images = {i: [] for i in range(10)}

        for img, label in self.train_data:
            if len(class_images[label]) < 300:  # begränsar för snabbhet
                class_images[label].append(img.squeeze())

        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i, ax in enumerate(axes.flat):
            avg_img = sum(class_images[i]) / len(class_images[i])
            ax.imshow(avg_img, cmap="gray")
            ax.set_title(f"Genomsnitt: {i}")
            ax.axis("off")

        plt.suptitle("Genomsnittlig bild per klass")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.show()
        print(f"Klassgenomsnitt sparade som: {save_path}")

    def run(self):
        self.summary()
        self.show_examples()
        self.show_class_averages()


if __name__ == "__main__":
    eda = MNISTEDA()
    eda.run()