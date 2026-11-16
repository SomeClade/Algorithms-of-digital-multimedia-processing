import cv2
import numpy as np
import matplotlib.pyplot as plt

if __package__:
    from .core import (finish_plots, load_image, otsu_threshold,
                       prepare_environment, save_comparison, save_images)
else:
    from core import (finish_plots, load_image, otsu_threshold,
                      prepare_environment, save_comparison, save_images)


def main():
    prepare_environment(__file__)

    original = load_image("images/rice.png", cv2.IMREAD_GRAYSCALE)


    threshold, binary = otsu_threshold(original)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    eroded = cv2.erode(binary, kernel)
    dilated = cv2.dilate(binary, kernel)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    removed = cv2.subtract(binary, opened)
    added = cv2.subtract(closed, opened)

    print("Порог Оцу:", threshold)
    print("Удалено пикселей при открытии:", np.count_nonzero(removed))
    print("Добавлено пикселей при закрытии:", np.count_nonzero(added))

    images = [original, binary, eroded, dilated, opened, closed]
    names = ["original", "binary", "eroded", "dilated", "opened", "closed"]
    titles = ["Фотография риса", "Бинаризация по Оцу", "Эрозия: эллипс 3 × 3",
              "Дилатация: эллипс 3 × 3", "Открытие: удаление выступов",
              "Закрытие после открытия"]

    save_images("task1", dict(zip(names, images)))
    save_comparison(images, titles, "results/task1_comparison.png",
                    layout=(2, 3), figsize=(12, 8), vmin=0, vmax=255)

    changes = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    changes[removed > 0] = [255, 0, 0]
    changes[added > 0] = [0, 180, 255]
    plt.figure(figsize=(12, 5))
    for index, image in enumerate([binary, closed, changes]):
        plt.subplot(1, 3, index + 1)
        plt.imshow(image[55:195, 115:270], cmap="gray", vmin=0, vmax=255,
                   interpolation="nearest")
        plt.title(["До обработки", "После обработки",
                   "Красный: удалено; голубой: добавлено"][index])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/task1_details.png", dpi=180)
    save_images("task1", {"removed": removed, "added": added})

    finish_plots()


if __name__ == "__main__":
    main()
