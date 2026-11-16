"""Задание 3. Сегментация фотографии методом K-means в плоскости a*b*."""

import os
import csv
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt


K = 4  # Три цветные грани кубика и нейтральная область (фон и черные ребра).
SEED = 42
MAX_DIMENSION = 640


def segment_colors(original, k=K, seed=SEED):
    """Возвращает float Lab, метки 0..k-1, центры a*b* и сумму квадратов ошибок."""
    # Для float BGR OpenCV ожидает 0..1 и возвращает L*=0..100, знаковые a*, b*.
    lab = cv2.cvtColor(original.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    samples = np.ascontiguousarray(lab[:, :, 1:3].reshape(-1, 2))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    cv2.setRNGSeed(seed)
    compactness, raw_labels, centers = cv2.kmeans(
        samples, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    # Нумерация кластеров не имеет смысла сама по себе: фиксируем порядок по a*, b*.
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    remap = np.empty(k, dtype=np.uint8)
    remap[order] = np.arange(k, dtype=np.uint8)
    labels = remap[raw_labels.ravel()].reshape(original.shape[:2])
    return lab, labels, centers[order], float(compactness)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("results", exist_ok=True)

    original = cv2.imread("images/color_objects.jpg")
    if original is None:
        raise FileNotFoundError("Не удалось прочитать images/color_objects.jpg")
    source_shape = list(original.shape[:2])
    # Уменьшаем весь снимок, сохраняя пропорции; кадрирование не используется.
    scale = min(1.0, MAX_DIMENSION / max(original.shape[:2]))
    if scale < 1.0:
        original = cv2.resize(
            original, (round(original.shape[1] * scale), round(original.shape[0] * scale)),
            interpolation=cv2.INTER_AREA
        )

    lab, labels, centers, compactness = segment_colors(original)
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    pixel_count = labels.size
    counts = np.bincount(labels.ravel(), minlength=K)
    mean_rgb = np.array([rgb[labels == i].mean(axis=0) for i in range(K)])
    palette = np.round(mean_rgb).astype(np.uint8)
    segmented_rgb = palette[labels]
    masks = [(labels == i).astype(np.uint8) * 255 for i in range(K)]
    masked = [cv2.bitwise_and(original, original, mask=mask) for mask in masks]

    output_images = {
        "original": original,
        "segmented": cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR),
    }
    for i in range(K):
        output_images[f"mask_{i}"] = masks[i]
        output_images[f"cluster_{i}"] = masked[i]
    for name, image in output_images.items():
        if not cv2.imwrite(f"results/task3_{name}.png", image):
            raise OSError(f"Не удалось сохранить results/task3_{name}.png")
    np.save("results/task3_labels.npy", labels)

    with open("results/task3_centers.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "a_star", "b_star", "pixels", "share", "mean_R", "mean_G", "mean_B"])
        for i in range(K):
            writer.writerow([i, float(centers[i, 0]), float(centers[i, 1]),
                             int(counts[i]), float(counts[i] / pixel_count),
                             *[float(value) for value in mean_rgb[i]]])

    metrics = {
        "method": "K-means on CIE Lab a*b*; L* is excluded from clustering",
        "k": K, "seed": SEED, "attempts": 10, "initialization": "KMEANS_PP_CENTERS",
        "criteria": {"max_iterations": 100, "epsilon": 0.2},
        "label_order": "lexicographic ascending a_star, then b_star",
        "source_shape_hw": source_shape, "working_shape_hw": list(labels.shape),
        "max_dimension": MAX_DIMENSION, "resize_scale": scale,
        "resize_interpolation": "INTER_AREA", "cropped": False,
        "lab_dtype": str(lab.dtype),
        "lab_min": lab.reshape(-1, 3).min(axis=0).tolist(),
        "lab_max": lab.reshape(-1, 3).max(axis=0).tolist(),
        "compactness_sse": compactness, "mean_squared_ab_distance": compactness / pixel_count,
        "cluster_pixels": counts.tolist(), "cluster_shares": (counts / pixel_count).tolist(),
        "source": {
            "title": "Rubik's Cube", "author": "William Warby", "license": "CC BY 2.0",
            "page": "https://commons.wikimedia.org/wiki/File:Rubiks_Cube_(11913013374).jpg",
            "original_page": "https://www.flickr.com/photos/wwarby/11913013374/",
            "license_url": "https://creativecommons.org/licenses/by/2.0/",
            "input": "Unaltered Wikimedia 960px JPEG thumbnail; resized here to 640px width"
        }
    }
    with open("results/task3_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    comparison = [rgb, segmented_rgb] + [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in masked]
    titles = [f"Исходная фотография ({original.shape[1]} × {original.shape[0]})",
              "K-means: средний RGB каждого кластера"]
    titles += [f"Кластер {i}: {counts[i] / pixel_count:.1%} пикселей" for i in range(K)]
    for ax, image, title in zip(axes.ravel(), comparison, titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Сегментация по цвету: K = 4, признаки a* и b*", fontsize=14)
    fig.tight_layout()
    fig.savefig("results/task3_comparison.png", dpi=150)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Исходная фотография")
    axes[0].axis("off")
    channel_specs = [("L* (0..100)", "gray", 0, 100),
                     ("a*: зеленый ← 0 → красный", "RdYlGn_r", -128, 127),
                     ("b*: синий ← 0 → желтый", "cividis", -128, 127)]
    for i, (title, cmap, low, high) in enumerate(channel_specs):
        image = axes[i + 1].imshow(lab[:, :, i], cmap=cmap, vmin=low, vmax=high)
        axes[i + 1].set_title(title, fontsize=10)
        axes[i + 1].axis("off")
        fig.colorbar(image, ax=axes[i + 1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig("results/task3_lab_channels.png", dpi=150)

    # Для графика берем не более 12 000 точек; K-means использует все пиксели.
    ab = lab[:, :, 1:3].reshape(-1, 2)
    sample_ids = np.random.default_rng(SEED).choice(pixel_count, min(12000, pixel_count), replace=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(ab[sample_ids, 0], ab[sample_ids, 1],
               c=segmented_rgb.reshape(-1, 3)[sample_ids] / 255.0,
               s=5, alpha=0.35, edgecolors="none")
    ax.scatter(centers[:, 0], centers[:, 1], c=palette / 255.0,
               s=180, marker="X", edgecolors="black", linewidths=1.2)
    for i, (a_value, b_value) in enumerate(centers):
        ax.annotate(str(i), (a_value, b_value), xytext=(8, 8), textcoords="offset points",
                    fontsize=12, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("a* (зеленый − / красный +)")
    ax.set_ylabel("b* (синий − / желтый +)")
    ax.set_title("Цветовые признаки a*b* и центры кластеров (X)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig("results/task3_ab_scatter.png", dpi=150)

    print("Task 3: CIE Lab a*b*, K=4, seed=42, attempts=10")
    print("Working image:", original.shape[1], "x", original.shape[0])
    for i in range(K):
        print(f"Cluster {i}: a*={centers[i, 0]:.3f}, b*={centers[i, 1]:.3f}, "
              f"pixels={counts[i]}, share={counts[i] / pixel_count:.4%}")
    print(f"Mean squared a*b* distance: {compactness / pixel_count:.6f}")
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
