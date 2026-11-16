import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


def skin_rgb(rgb):
    """Критерий (4.12) для RGB в диапазоне 0..255."""
    red, green, blue = np.moveaxis(rgb.astype(np.int16), -1, 0)
    # B < 20 напечатано в методичке. Условие сохранено без исправления.
    selected = ((red > 95) & (green > 40) & (blue < 20)
                & (rgb.max(axis=2).astype(np.int16)
                   - rgb.min(axis=2).astype(np.int16) > 15)
                & (np.abs(red - green) > 15) & (red > green) & (red > blue))
    return selected.astype(np.uint8) * 255


def skin_bright(rgb):
    """Критерий (4.13) для ярко освещенной кожи."""
    red, green, blue = np.moveaxis(rgb.astype(np.int16), -1, 0)
    selected = ((red > 220) & (green > 210) & (blue > 170)
                & (np.abs(red - green) <= 15) & (green > blue) & (red > blue))
    return selected.astype(np.uint8) * 255


def skin_normalized(rgb):
    """Критерий (4.14); черные пиксели и G=0 не дают деления на ноль."""
    values = rgb.astype(np.float64)
    total = values.sum(axis=2, keepdims=True)
    normalized = np.divide(values, total, out=np.zeros_like(values), where=total > 0)
    red, green, blue = np.moveaxis(normalized, -1, 0)
    denominator = (red + green + blue) ** 2
    red_green = np.divide(red, green, out=np.zeros_like(red), where=green > 0)
    red_blue_product = np.divide(red * blue, denominator,
                                 out=np.zeros_like(red), where=denominator > 0)
    red_green_product = np.divide(red * green, denominator,
                                  out=np.zeros_like(red), where=denominator > 0)
    selected = ((red_green > 1.185) & (red_blue_product > 0.107)
                & (red_green_product > 0.112))
    return selected.astype(np.uint8) * 255



def skin_rgb_corrected(rgb):
    """RGB с B > 20, проверенное по Kovač, Peer, Solina (2003)."""
    red, green, blue = np.moveaxis(rgb.astype(np.int16), -1, 0)
    selected = ((red > 95) & (green > 40) & (blue > 20)
                & (rgb.max(axis=2).astype(np.int16)
                   - rgb.min(axis=2).astype(np.int16) > 15)
                & (np.abs(red - green) > 15) & (red > green) & (red > blue))
    return selected.astype(np.uint8) * 255


def segment_face(rgb):
    # Объединяем правила, дополняющие друг друга при разном освещении.
    corrected = skin_rgb_corrected(rgb)
    bright = skin_bright(rgb)
    normalized = skin_normalized(rgb)
    raw = corrected | bright | normalized
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)

    # Отбор ДО закрытия не дает разрозненному фону приклеиться к лицу.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    largest = np.zeros_like(raw)
    if count > 1:
        label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest[labels == label] = 255
    closed = cv2.morphologyEx(largest, cv2.MORPH_CLOSE, kernel)

    # Цельная область лица/шеи: глаза и очки внутри контура сохраняются.
    # Это регион по цвету кожи, а не маска только чистых пикселей кожи.
    padded = cv2.copyMakeBorder(closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    cv2.floodFill(flooded, None, (0, 0), 255)
    clean = closed | cv2.bitwise_not(flooded)[1:-1, 1:-1]
    return {"rgb_corrected_raw": corrected, "bright_raw": bright,
            "normalized_raw": normalized, "combined_raw": raw, "opened": opened,
            "largest": largest, "closed": closed, "clean": clean}


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("results", exist_ok=True)
    original = cv2.imread("images/portrait.jpg")
    if original is None:
        raise FileNotFoundError("Не удалось прочитать images/portrait.jpg")
    height, width = original.shape[:2]
    scale = min(1.0, 640 / max(height, width))
    image = cv2.resize(original, (round(width * scale), round(height * scale)),
                       interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    source_mask = skin_rgb(rgb)
    stages = segment_face(rgb)
    clean = stages["clean"]
    overlay = image.copy()
    green = np.full_like(image, (0, 255, 0))
    blended = cv2.addWeighted(image, 0.6, green, 0.4, 0)
    overlay[clean > 0] = blended[clean > 0]
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    extracted = cv2.bitwise_and(image, image, mask=clean)
    outputs = {"original": image, "rgb_raw": source_mask, **stages,
               "overlay": overlay, "extracted": extracted}
    for name, stage in outputs.items():
        cv2.imwrite("results/task2_" + name + ".png", stage)

    plt.figure(figsize=(12, 11))
    shown_images = [rgb, source_mask, stages["rgb_corrected_raw"], stages["bright_raw"],
                    stages["normalized_raw"], stages["combined_raw"]]
    titles = ["Исходное фото", "RGB: B < 20 — как напечатано",
              "RGB: B > 20 — исправленный знак", "Яркое освещение (4.13)",
              "Нормированные RGB (4.14)", "Объединение трёх рабочих критериев"]
    for index, shown in enumerate(shown_images):
        plt.subplot(2, 3, index + 1)
        plt.imshow(shown, cmap="gray", vmin=0, vmax=255)
        plt.title(titles[index], fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/task2_criteria_comparison.png", dpi=150)

    plt.figure(figsize=(12, 11))
    shown_images = [rgb, stages["combined_raw"], stages["largest"], clean,
                    cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)]
    titles = ["Исходное фото", "Объединённая цветовая маска",
              "Открытие 5 × 5 → крупнейшая область", "Закрытие и заполнение отверстий",
              "Наложение области лица/шеи", "Выделенная область лица/шеи"]
    for index, shown in enumerate(shown_images):
        plt.subplot(2, 3, index + 1)
        plt.imshow(shown, cmap="gray", vmin=0, vmax=255)
        plt.title(titles[index], fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/task2_comparison.png", dpi=150)

    previous = cv2.imread("results/task2_previous_extracted.png")
    if previous is not None and previous.shape == image.shape:
        plt.figure(figsize=(11, 6))
        for index, (shown, title) in enumerate(zip([image, previous, extracted],
                ["Исходное фото", "Прежний результат: пропущена щека", "Исправленный результат"])):
            plt.subplot(1, 3, index + 1)
            plt.imshow(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
            plt.title(title, fontsize=10)
            plt.axis("off")
        plt.tight_layout()
        plt.savefig("results/task2_before_after.png", dpi=150)

    pixel_count = clean.size
    counts = {name: int(np.count_nonzero(mask)) for name, mask in
              {"rgb_raw": source_mask, **stages}.items()}
    statistics = {
        "source": "images/portrait.jpg",
        "source_size": {"width": width, "height": height},
        "processed_size": {"width": image.shape[1], "height": image.shape[0]},
        "selected_criterion": "corrected_RGB_OR_bright_OR_normalized",
        "ordinary_rgb_blue_rule": "B > 20; source typo B < 20 retained separately",
        "opening_ellipse": [5, 5], "closing_ellipse": [5, 5],
        "selection_order": "opening -> largest component -> closing -> fill holes",
        "connectivity": 8,
        "kept_components": int(np.any(clean)), "pixel_count": int(pixel_count),
        "skin_pixels": counts,
        "skin_percent": {name: round(100 * count / pixel_count, 3) for name, count in counts.items()},
        "removed_disconnected_pixels": int(np.count_nonzero(stages["opened"])
                                           - np.count_nonzero(stages["largest"])),
        "filled_interior_pixels": int(np.count_nonzero(clean) - np.count_nonzero(stages["closed"])),
        "output_meaning": "Face/neck region based on skin color, including eyes/glasses/lips inside.",
        "limitations": "Цветовые правила не дают точного семантического контура кожи.",
        "reference": "https://citeseerx.ist.psu.edu/document?doi=05cdcec7ff0d97049b7502cfbeb3dc747f97dddd&repid=rep1&type=pdf"
    }
    with open("results/task2_stats.json", "w", encoding="utf-8") as file:
        json.dump(statistics, file, ensure_ascii=False, indent=2)
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
