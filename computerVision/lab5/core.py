import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def prepare_environment(script_path):
    os.chdir(os.path.dirname(os.path.abspath(script_path)))
    os.makedirs("results", exist_ok=True)


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать {path}")
    return image


def resize_image(image, max_dimension=640):
    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        return cv2.resize(image, (round(width * scale), round(height * scale)),
                          interpolation=cv2.INTER_AREA)
    return image


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def finish_plots():
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close("all")


def local_mean_std(image, window):
    values = image.astype(np.float32)
    mean = cv2.boxFilter(values, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    mean_square = cv2.boxFilter(values * values, -1, (window, window),
                                borderType=cv2.BORDER_REFLECT)
    deviation = np.sqrt(np.maximum(mean_square - mean * mean, 0))
    return mean, deviation


def fill_holes(mask):
    """Заполняет внутренние отверстия бинарной маски, сохраняя внешний фон."""
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    cv2.floodFill(flooded, None, (0, 0), 255)
    holes = cv2.bitwise_not(flooded)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)
