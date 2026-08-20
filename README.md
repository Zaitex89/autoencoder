# autoencoder


"""
A simple autoencoder trained on the MNIST dataset of handwritten digits.
 
An autoencoder learns to compress data into a small "latent" code and then
reconstruct the original from that code. It trains WITHOUT labels: the target
output for each image is just the image itself, so the network is forced to
learn a compact representation that keeps the important information.
 
Requirements:
    pip install torch torchvision matplotlib
 
Run:
    python autoencoder_mnist.py
"""