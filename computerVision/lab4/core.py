import os

import cv2
import matplotlib.pyplot as plt


def prepare_environment(script_path):
    os.chdir(os.path.dirname(os.path.abspath(script_path)))
    os.makedirs("results", exist_ok=True)


def load_image(path, flags=cv2.IMREAD_COLOR):
    image = cv2.imread(path, flags)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать {path}")
    return image


def otsu_threshold(image, inverse=False):
    mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    return cv2.threshold(image, 0, 255, mode + cv2.THRESH_OTSU)


def save_images(prefix, images):
    for name, image in images.items():
        cv2.imwrite(f"results/{prefix}_{name}.png", image)


def save_comparison(images, titles, path, layout, figsize, cmaps=None,
                    vmin=None, vmax=None):
    """Сохраняет сетку сравнения с отдельными палитрами и общим диапазоном."""
    plt.figure(figsize=figsize)
    for index, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(*layout, index + 1)
        cmap = "gray" if cmaps is None else cmaps[index]
        plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)


def finish_plots():
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close("all")
