import cv2
import numpy as np

if __package__:
    from .core import (finish_plots, load_image, otsu_threshold,
                       prepare_environment, save_comparison, save_images)
else:
    from core import (finish_plots, load_image, otsu_threshold,
                      prepare_environment, save_comparison, save_images)


def main():
    prepare_environment(__file__)

    original = load_image("images/coins.jpg")

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = otsu_threshold(smoothed, inverse=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Положительный distanceTransform имеет максимумы в центрах монет.
    # У отрицательной карты эти же точки являются локальными минимумами.
    distance = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    _, foreground = cv2.threshold(distance, 0.7 * distance.max(), 255, cv2.THRESH_BINARY)
    foreground = foreground.astype(np.uint8)

    # expanded — область возможного объекта, а ее дополнение — уверенный фон.
    expanded = cv2.dilate(opened, kernel, iterations=3)
    background = cv2.bitwise_not(expanded)
    unknown = cv2.subtract(expanded, foreground)

    marker_count, markers = cv2.connectedComponents(foreground)
    markers = markers + 1  # Фон: 1; монеты: 2, 3, ...
    markers[unknown > 0] = 0
    initial_markers = markers.copy()
    cv2.watershed(cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR), markers)

    object_labels = np.unique(markers[markers > 1])
    print("Маркеров объектов:", marker_count - 1)
    print("Областей после водораздела:", len(object_labels))

    colors = np.random.default_rng(42).integers(50, 256, (marker_count + 1, 3),
                                               dtype=np.uint8)
    segmented = np.zeros_like(original)
    for label in object_labels:
        segmented[markers == label] = colors[label]
    segmented[markers == -1] = [255, 255, 255]
    overlay = original.copy()
    overlay[markers == -1] = [0, 0, 255]

    save_images("task3", {"binary": binary, "opened": opened, "foreground": foreground,
                          "background": background, "unknown": unknown,
                          "segmented": segmented, "overlay": overlay})
    np.save("results/task3_distance.npy", distance)
    np.save("results/task3_initial_markers.npy", initial_markers)
    np.save("results/task3_labels.npy", markers)

    images = [cv2.cvtColor(original, cv2.COLOR_BGR2RGB), opened, distance,
              foreground, initial_markers, cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB),
              unknown, background, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)]
    titles = ["Исходная фотография", "Маска после открытия", "Карта расстояний",
              "Уверенные центры монет", "Маркеры: 0 — неизвестно, 1 — фон",
              "Результат сегментации", "Неизвестная область", "Уверенный фон",
              "Границы водоразделов"]

    cmaps = ["nipy_spectral" if index == 4 else "gray" for index in range(len(images))]
    save_comparison(images, titles, "results/task3_comparison.png",
                    layout=(3, 3), figsize=(12, 13), cmaps=cmaps)

    finish_plots()


if __name__ == "__main__":
    main()
