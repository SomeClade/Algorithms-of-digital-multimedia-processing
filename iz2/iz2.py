import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

IMAGE_FOLDER = 'dataset/'

IMAGE_NAMES = [f"{i}.jpg" for i in range(1, 10)]

GAUSSIAN_KERNEL_SIZES = [(3, 3), (5, 5), (7, 7)]

# Пары пороговых значений для алгоритма Канни
CANNY_THRESHOLD_PAIRS = [
    (50, 150),
    (100, 200),
    (150, 250)
]

OPERATORS = {
    'Sobel': None,  # Оператор Собеля используется по умолчанию
    'Scharr': cv2.CV_64F  # Оператор Шарра
}

# Альтернативные методы выявления границ
ALTERNATIVE_METHODS = {
    'Laplacian': None,
    'Prewitt': None,
    'Zero_Crossing': None,
    'Difference_of_Gaussians': None
}

def load_images(folder, image_names):
    images = []
    valid_image_names = []
    for filename in image_names:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            valid_image_names.append(filename)
        else:
            print(f"⚠️ Не удалось загрузить изображение: {filename}")
    return images, valid_image_names

def apply_gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_canny(image, lower_thresh, upper_thresh, operator='Sobel'):
    if operator == 'Scharr':
        # Используем оператор Шарра вместо Собеля
        grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        edges = cv2.Canny(grad, lower_thresh, upper_thresh)
        return edges
    else:
        # Используем стандартный алгоритм Канни с оператором Собеля
        return cv2.Canny(image, lower_thresh, upper_thresh)

# Функция для применения альтернативных методов
def apply_alternative_method(image, method):
    if method == 'Laplacian':
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        edges = cv2.convertScaleAbs(laplacian)
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
        return edges
    elif method == 'Prewitt':
        # Определение ядер Прюитта
        kernelx = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]], dtype=int)
        kernely = np.array([[1, 1, 1],
                            [0, 0, 0],
                            [-1, -1, -1]], dtype=int)
        grad_x = cv2.filter2D(image, cv2.CV_64F, kernelx)
        grad_y = cv2.filter2D(image, cv2.CV_64F, kernely)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        _, edges = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)
        return edges
    elif method == 'Roberts':
        # Определение ядер Робертса
        kernelx = np.array([[1, 0],
                            [0, -1]], dtype=int)
        kernely = np.array([[0, 1],
                            [-1, 0]], dtype=int)
        grad_x = cv2.filter2D(image, cv2.CV_64F, kernelx)
        grad_y = cv2.filter2D(image, cv2.CV_64F, kernely)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        _, edges = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)
        return edges
    elif method == 'Zero_Crossing':
        # Использование Zero Crossing на основе Лапласиана Гаусса
        blurred = cv2.GaussianBlur(image, (3,3), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        zero_cross = np.zeros_like(laplacian, dtype=np.uint8)
        # Определение нулевых переходов
        zero_cross[(laplacian > 0) & (cv2.Laplacian(blurred, cv2.CV_64F, ksize=3) < 0)] = 255
        zero_cross[(laplacian < 0) & (cv2.Laplacian(blurred, cv2.CV_64F, ksize=3) > 0)] = 255
        return zero_cross
    elif method == 'Difference_of_Gaussians':
        # Разница Гауссианов
        blur1 = cv2.GaussianBlur(image, (3,3), 0)
        blur2 = cv2.GaussianBlur(image, (5,5), 0)
        dog = cv2.subtract(blur1, blur2)
        _, edges = cv2.threshold(dog, 10, 255, cv2.THRESH_BINARY)
        return edges
    else:
        return None

def display_comparison(original, results, image_name, save_path=None):
    num_methods = len(results)
    cols = 3  # Количество столбцов в сетке
    rows = (num_methods + 1) // cols + 1  # Количество строк, включая исходное изображение

    plt.figure(figsize=(5 * cols, 5 * rows))

    plt.subplot(rows, cols, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Исходное')
    plt.axis('off')

    for idx, result in enumerate(results):
        plt.subplot(rows, cols, idx + 2)
        plt.imshow(result['Edges'], cmap='gray')
        title = f"{result['Method']}"
        if result['Method'].startswith('Canny'):
            title += f"\nOp: {result['Operator']}\nKernel: {result['Gaussian Kernel']}\nThresh: {result['Lower Threshold']}-{result['Upper Threshold']}"
        elif result['Method'] in ['Zero_Crossing', 'Difference_of_Gaussians']:
            title += f"\nMethod: {result['Method']}"
        plt.title(title, fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

images, image_names = load_images(IMAGE_FOLDER, IMAGE_NAMES)
print(f"✅ Загружено {len(images)} изображений.")

all_results = []
optimal_results = []

VISUALIZATION_FOLDER = 'visualizations/'
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

# Основной цикл обработки изображений
for idx, image in enumerate(images):
    image_name = image_names[idx]
    print(f"\n🖼 Обработка изображения: {image_name}")

    # Для хранения оптимальных параметров
    max_edges = -1
    optimal_method = ''
    optimal_operator = ''
    optimal_kernel = ()
    optimal_thresholds = ()

    # Список для визуализации всех методов для текущего изображения
    comparison_results = []

    # Применение методов Канни с различными параметрами
    for operator_name, operator in OPERATORS.items():
        for kernel_size in GAUSSIAN_KERNEL_SIZES:
            blurred = apply_gaussian_blur(image, kernel_size)
            for thresh_pair in CANNY_THRESHOLD_PAIRS:
                lower, upper = thresh_pair
                edges = apply_canny(blurred, lower, upper, operator=operator_name)
                num_edges = np.sum(edges > 0)

                all_results.append({
                    'Image': image_name,
                    'Method': 'Canny',
                    'Operator': operator_name,
                    'Gaussian Kernel': kernel_size,
                    'Lower Threshold': lower,
                    'Upper Threshold': upper,
                    'Num Edges': num_edges
                })

                comparison_results.append({
                    'Method': f'Canny ({operator_name})',
                    'Operator': operator_name,
                    'Gaussian Kernel': kernel_size,
                    'Lower Threshold': lower,
                    'Upper Threshold': upper,
                    'Edges': edges
                })

                if num_edges > max_edges:
                    max_edges = num_edges
                    optimal_method = 'Canny'
                    optimal_operator = operator_name
                    optimal_kernel = kernel_size
                    optimal_thresholds = (lower, upper)

    # Применение альтернативных методов
    for method_name in ALTERNATIVE_METHODS.keys():
        edges = apply_alternative_method(image, method_name)
        if edges is not None:
            num_edges = np.sum(edges > 0)

            # Сохранение всех результатов
            all_results.append({
                'Image': image_name,
                'Method': method_name,
                'Operator': 'N/A',
                'Gaussian Kernel': 'N/A',
                'Lower Threshold': 'N/A',
                'Upper Threshold': 'N/A',
                'Num Edges': num_edges
            })

            comparison_results.append({
                'Method': method_name,
                'Operator': 'N/A',
                'Gaussian Kernel': 'N/A',
                'Lower Threshold': 'N/A',
                'Upper Threshold': 'N/A',
                'Edges': edges
            })

            if num_edges > max_edges:
                max_edges = num_edges
                optimal_method = method_name
                optimal_operator = 'N/A'
                optimal_kernel = 'N/A'
                optimal_thresholds = ('N/A', 'N/A')

    optimal_results.append({
        'Image': image_name,
        'Method': optimal_method,
        'Operator': optimal_operator,
        'Gaussian Kernel': optimal_kernel,
        'Lower Threshold': optimal_thresholds[0],
        'Upper Threshold': optimal_thresholds[1],
        'Num Edges': max_edges
    })

    save_path = os.path.join(VISUALIZATION_FOLDER, f"{os.path.splitext(image_name)[0]}_comparison.png")
    display_comparison(image, comparison_results, image_name, save_path=save_path)

df_all = pd.DataFrame(all_results)

df_optimal = pd.DataFrame(optimal_results)

df_all.to_csv('edge_detection_results.csv', index=False, encoding='utf-8-sig')
print("\n📊 Все результаты сохранены в 'edge_detection_results.csv'.")

df_optimal.to_csv('optimal_edge_detection_results.csv', index=False, encoding='utf-8-sig')
print("✅ Оптимальные результаты сохранены в 'optimal_edge_detection_results.csv'.")

best_method_overall = df_optimal['Method'].mode()[0]
print(f"\n🏆 Лучший метод по всем изображениям: {best_method_overall}")

with open('best_method_overall.txt', 'w', encoding='utf-8') as f:
    f.write(f"🏆 Лучший метод по всем изображениям: {best_method_overall}\n")
print("📄 Лучший метод сохранён в 'best_method_overall.txt'.")
