import numpy as np

def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel

if __name__ == "__main__":
    sizes = [3, 5, 7]
    sigma = 1.0
    for size in sizes:
        kernel = gaussian_kernel(size, sigma)
        print(f"Gaussian Kernel (size={size}, sigma={sigma}):\n", kernel, "\n")
