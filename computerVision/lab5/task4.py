import csv
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __package__:
    from .core import (fill_holes, finish_plots, load_image, local_mean_std,
                       prepare_environment, resize_image, save_json)
else:
    from core import (fill_holes, finish_plots, load_image, local_mean_std,
                      prepare_environment, resize_image, save_json)


def local_entropy(gray, window=9, bins=32):
    quantized = gray.astype(np.uint16) * bins // 256
    entropy = np.zeros(gray.shape, dtype=np.float32)
    for level in range(bins):
        indicator = (quantized == level).astype(np.float32)
        probability = cv2.boxFilter(indicator, -1, (window, window),
                                    borderType=cv2.BORDER_REFLECT)
        selected = probability > 0
        entropy[selected] -= probability[selected] * np.log2(probability[selected])
    return np.maximum(entropy, 0)


def local_statistics(gray, window=9):
    _, deviation = local_mean_std(gray, window)
    kernel = np.ones((window, window), dtype=np.uint8)
    maximum = cv2.dilate(gray, kernel, borderType=cv2.BORDER_REFLECT)
    minimum = cv2.erode(gray, kernel, borderType=cv2.BORDER_REFLECT)
    return deviation, maximum - minimum


def texture_descriptors(gray, mask):
    values = gray[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        raise ValueError("Нельзя вычислить признаки пустой области")
    histogram = np.bincount(values, minlength=256).astype(np.float64)
    probability = histogram / values.size
    levels = np.arange(256, dtype=np.float64)
    mean = np.sum(levels * probability)
    variance = np.sum((levels - mean) ** 2 * probability)
    positive = probability[probability > 0]
    third_moment = np.sum((levels - mean) ** 3 * probability)
    return {"mean": float(mean), "std": float(np.sqrt(variance)),
            "smoothness": float(1 - 1 / (1 + variance / 255 ** 2)),
            "third_moment": float(third_moment),
            "third_moment_normalized": float(third_moment / 255 ** 3),
            "uniformity": float(np.sum(probability ** 2)),
            "entropy": float(-np.sum(positive * np.log2(positive)))}


def remove_small_regions(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            result[labels == label] = 255
    return result


def main():
    prepare_environment(__file__)
    original = load_image("images/textures.jpg")
    height, width = original.shape[:2]
    image = resize_image(original)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    entropy = local_entropy(gray, window=9, bins=32)
    deviation, value_range = local_statistics(gray, window=9)
    normalized = cv2.normalize(entropy, None, 0, 255, cv2.NORM_MINMAX)
    entropy_image = np.rint(normalized).astype(np.uint8)
    threshold, raw = cv2.threshold(entropy_image, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Аналог bwareaopen с учетом размера рабочей фотографии.
    min_area = max(1, round(gray.size * 0.005))
    filtered = remove_small_regions(raw, min_area)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    coarse = fill_holes(closed)
    smooth = cv2.bitwise_not(coarse)
    masks = [smooth, coarse]
    class_names = ["smooth_sky", "coarse_field"]
    class_titles = ["Гладкая текстура: небо", "Грубая текстура: поле"]
    labels = (coarse > 0).astype(np.uint8)

    boundary = cv2.subtract(coarse, cv2.erode(coarse, np.ones((3, 3), np.uint8)))
    overlay = image.copy()
    overlay[boundary > 0] = [0, 0, 255]
    extracted = [cv2.bitwise_and(image, image, mask=mask) for mask in masks]
    for name, result in zip(["original", "gray", "entropy", "raw", "area_filtered",
                             "closed", "smooth", "coarse", "boundary", "overlay",
                             "smooth_texture", "coarse_texture"],
                            [image, gray, entropy_image, raw, filtered, closed,
                             smooth, coarse, boundary, overlay, *extracted]):
        cv2.imwrite("results/task4_" + name + ".png", result)
    np.save("results/task4_labels.npy", labels)
    np.save("results/task4_entropy.npy", entropy)
    np.save("results/task4_local_std.npy", deviation)
    np.save("results/task4_local_range.npy", value_range)

    records = []
    for name, mask in zip(class_names, masks):
        selected = mask > 0
        record = {"class": name, "pixels": int(np.count_nonzero(selected)),
                  "fraction": float(np.mean(selected)), **texture_descriptors(gray, mask),
                  "local_entropy_mean": float(np.mean(entropy[selected])),
                  "local_std_mean": float(np.mean(deviation[selected])),
                  "local_range_mean": float(np.mean(value_range[selected]))}
        records.append(record)
    with open("results/task4_metrics.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    statistics = {"source": "images/textures.jpg", "source_size": [width, height],
                  "processed_size": [image.shape[1], image.shape[0]], "window": 9,
                  "local_entropy_bins": 32, "regional_histogram_bins": 256,
                  "otsu_normalized_threshold": float(threshold),
                  "entropy_min": float(entropy.min()), "entropy_max": float(entropy.max()),
                  "min_region_area": min_area, "closing_ellipse": [9, 9],
                  "class_assignment": "Higher local variation: coarse field; lower: smooth sky."
                  " Relative interpretation for this photograph, not universal trained classification.",
                  "regions": records}
    save_json("results/task4_stats.json", statistics)

    plt.figure(figsize=(13, 8))
    maps = [rgb, gray, entropy, deviation, value_range, raw]
    titles = ["Исходная фотография", "Яркость", "Локальная энтропия: 9 × 9, 32 уровня",
              "Локальное стандартное отклонение", "Локальный размах: max − min",
              "Оцу по энтропии: T = " + str(round(threshold))]
    for index, (shown, title) in enumerate(zip(maps, titles)):
        plt.subplot(2, 3, index + 1)
        plt.imshow(shown, cmap="magma" if index in [2, 3, 4] else "gray")
        plt.title(title, fontsize=10)
        plt.axis("off")
        if index in [2, 3, 4]:
            plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("results/task4_features.png", dpi=150)

    plt.figure(figsize=(13, 8))
    maps = [rgb, coarse, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), smooth,
            *[cv2.cvtColor(part, cv2.COLOR_BGR2RGB) for part in extracted]]
    titles = ["Исходная фотография", "Грубая текстура после обработки",
              "Граница сегментов", "Гладкая текстура: дополнение маски", *class_titles]
    for index, (shown, title) in enumerate(zip(maps, titles)):
        plt.subplot(2, 3, index + 1)
        plt.imshow(shown, cmap="gray", vmin=0, vmax=255)
        plt.title(title, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/task4_comparison.png", dpi=150)

    print("Порог Оцу нормированной энтропии:", threshold)
    for record in records:
        print(json.dumps(record, ensure_ascii=False))
    finish_plots()


if __name__ == "__main__":
    main()
