import cv2
from task3_gaussian_filter import apply_gaussian_filter_manual


def compare_gaussian_filters(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Параметры фильтра
    params = [(3, 10.0), (3, 200.0), (5, 10.0), (5, 200.0)]

    for kernel_size, sigma in params:
        filtered_img = apply_gaussian_filter_manual(img, kernel_size, sigma)
        output_filename = f"filtered_manual_{kernel_size}_{sigma}.jpg"
        cv2.imwrite(output_filename, filtered_img)
        print(
            f"Фильтр Гаусса применен с размером {kernel_size} и sigma {sigma}. Результат сохранен в {output_filename}")


if __name__ == "__main__":
    compare_gaussian_filters('images/img1.jpg')
