import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __package__:
    from .core import (finish_plots, load_image, local_mean_std,
                       prepare_environment, resize_image, save_json)
else:
    from core import (finish_plots, load_image, local_mean_std,
                      prepare_environment, resize_image, save_json)


def double_threshold(image, lower=110, upper=200):
    return np.where((image > lower) & (image <= upper), 255, 0).astype(np.uint8)


def niblack_threshold(image, window=31, k=-0.2):
    values = image.astype(np.float32)
    mean, deviation = local_mean_std(image, window)
    threshold = mean + k * deviation
    return np.where(values > threshold, 255, 0).astype(np.uint8)


def main():
    prepare_environment(__file__)
    original = load_image("images/portrait.jpg")
    height, width = original.shape[:2]
    original = resize_image(original)
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    otsu_value, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks = [double_threshold(gray), otsu,
             cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 31, 5),
             cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 5),
             cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
             cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1],
             niblack_threshold(gray)]
    names = ["double", "otsu", "adaptive_mean", "adaptive_gaussian", "upper",
             "lower", "niblack"]
    titles = ["Двойной: 110 < I ≤ 200", "Оцу: T = " + str(round(otsu_value)),
              "Адаптивный: среднее, 31 / 5", "Адаптивный: Гаусс, 31 / 5",
              "Верхний: I > 127", "Нижний: I ≤ 127", "Ниблэк: 31, k = −0.2"]
    measurements = []
    plt.figure(figsize=(12, 13))
    for index, image in enumerate([rgb, gray] + masks):
        plt.subplot(3, 3, index + 1)
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        plt.title((["Исходное ЦВЕТНОЕ фото", "Яркость rgb2gray"] + titles)[index])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/task1_comparison.png", dpi=150)
    cv2.imwrite("results/task1_gray.png", gray)
    for name, mask in zip(names, masks):
        cv2.imwrite("results/task1_" + name + ".png", mask)
        measurements.append([name, round(float(np.mean(mask > 0)), 6)])

    # Дополнительно обрабатываем именно цветовые компоненты R/G/B.
    # AND — отдельное явно выбранное правило объединения; каждая маска бинарная.
    channel_otsu = []
    color_masks = []
    plt.figure(figsize=(12, 11))
    for method in range(3):
        per_channel = []
        for channel in range(3):
            values = rgb[:, :, channel]
            if method == 0:
                mask = double_threshold(values)
            elif method == 1:
                value, mask = cv2.threshold(values, 0, 255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                channel_otsu.append(float(value))
            else:
                mask = cv2.adaptiveThreshold(values, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 31, 5)
            per_channel.append(mask)
            cv2.imwrite("results/task1_rgb_" + ["double", "otsu", "adaptive"][method]
                        + "_" + "RGB"[channel] + ".png", mask)
        combined = np.bitwise_and.reduce(per_channel)
        color_masks.append(combined)
        cv2.imwrite("results/task1_rgb_" + ["double", "otsu", "adaptive"][method]
                    + "_and.png", combined)
        for channel, mask in enumerate(per_channel + [combined]):
            plt.subplot(3, 4, method * 4 + channel + 1)
            plt.imshow(mask, cmap="gray", vmin=0, vmax=255)
            plt.title(["Двойной", "Оцу", "Адаптивный Гаусс"][method] + ": "
                      + ["R", "G", "B", "R AND G AND B"][channel])
            plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/task1_rgb_comparison.png", dpi=150)

    plt.figure(figsize=(10, 4))
    for channel, color in enumerate(["red", "green", "blue"]):
        histogram = np.bincount(rgb[:, :, channel].ravel(), minlength=256)
        plt.plot(histogram, color=color, label="RGB"[channel])
    plt.axvline(110, color="black", linestyle="--", label="110 / 200")
    plt.axvline(200, color="black", linestyle="--")
    plt.axvline(otsu_value, color="purple", linestyle=":", label="Оцу по яркости")
    plt.xlabel("Интенсивность")
    plt.ylabel("Число пикселей")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/task1_histogram.png", dpi=150)

    with open("results/task1_metrics.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "white_fraction"])
        writer.writerows(measurements)
    stats = {"source": "portrait.jpg", "original_size": [width, height],
             "processed_size": [original.shape[1], original.shape[0]],
             "double_threshold": {"lower_strict": 110, "upper_inclusive": 200},
             "otsu_gray": float(otsu_value), "otsu_rgb": channel_otsu,
             "adaptive_window": 31, "adaptive_C": 5,
             "rgb_combination": "AND", "niblack_k": -0.2}
    save_json("results/task1_stats.json", stats)
    print("Порог Оцу по яркости:", otsu_value)
    print("Пороги Оцу по R/G/B:", channel_otsu)
    for row in measurements:
        print(row[0], ": доля белых пикселей", row[1])
    finish_plots()


if __name__ == "__main__":
    main()
