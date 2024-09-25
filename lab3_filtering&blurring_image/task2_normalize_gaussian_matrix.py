import numpy as np
from task1_gaussian_matrix import gaussian_kernel

def normalize_kernel(kernel):
    return kernel / np.sum(kernel)

if __name__ == "__main__":
    sizes = [3, 5, 7]
    sigma = 1.0
    for size in sizes:
        kernel = gaussian_kernel(size, sigma)
        normalized_kernel = normalize_kernel(kernel)
        print(f"Normalized Gaussian Kernel (size={size}, sigma={sigma}):\n", normalized_kernel, "\n")
