import cv2
from task3_gaussian_filter import apply_gaussian_filter_manual


def apply_gaussian_filter_opencv(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def compare_manual_vs_opencv(image_path, kernel_size, sigma):
    # Загружаем изображение в оттенках серого
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    manual_filtered_img = apply_gaussian_filter_manual(img, kernel_size, sigma)

    # Применяем размытие Гаусса встроенным методом OpenCV
    opencv_filtered_img = apply_gaussian_filter_opencv(img, kernel_size, sigma)

    # Сохраняем результаты
    cv2.imwrite(f'filtered_manual_{kernel_size}_{sigma}.jpg', manual_filtered_img)
    cv2.imwrite(f'filtered_opencv_{kernel_size}_{sigma}.jpg', opencv_filtered_img)

    print(f"Результаты сохранены: manual и OpenCV размытие для kernel_size={kernel_size} и sigma={sigma}.")


if __name__ == "__main__":
    compare_manual_vs_opencv('images/img1.jpg', 7, 20)
