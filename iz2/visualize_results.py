import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# Пути к файлам
EDGE_DETECTION_RESULTS = 'edge_detection_results.csv'
OPTIMAL_RESULTS = 'optimal_edge_detection_results.csv'
BEST_METHOD_FILE = 'best_method_overall.txt'
VISUALIZATION_FOLDER = 'visualizations/'

os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)


def read_data():
    """
    Чтение CSV и TXT файлов.
    """
    # Чтение всех результатов
    df_all = pd.read_csv(EDGE_DETECTION_RESULTS)
    print(f"✅ Считаны данные из '{EDGE_DETECTION_RESULTS}'.")

    # Чтение оптимальных результатов
    df_optimal = pd.read_csv(OPTIMAL_RESULTS)
    print(f"✅ Считаны данные из '{OPTIMAL_RESULTS}'.")

    # Чтение лучшего метода
    with open(BEST_METHOD_FILE, 'r', encoding='utf-8') as f:
        best_method = f.read().strip()
    print(f"✅ Считан лучший метод из '{BEST_METHOD_FILE}': {best_method}")

    return df_all, df_optimal, best_method


def visualize_num_edges_per_method(df_all):
    """
    Визуализация распределения количества обнаруженных границ по методам.
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Method', y='Num Edges', data=df_all)
    plt.title('Распределение количества обнаруженных границ по методам')
    plt.xlabel('Метод')
    plt.ylabel('Количество обнаруженных границ')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = os.path.join(VISUALIZATION_FOLDER, 'num_edges_per_method.png')
    plt.savefig(save_path)
    plt.show()
    print(f"📊 Визуализация распределения количества границ по методам сохранена в '{save_path}'.")


def visualize_optimal_methods(df_optimal):
    """
    Визуализация распределения оптимальных методов по изображениям.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Method', data=df_optimal)
    plt.title('Распределение оптимальных методов по изображениям')
    plt.xlabel('Оптимальный метод')
    plt.ylabel('Количество изображений')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = os.path.join(VISUALIZATION_FOLDER, 'optimal_methods_distribution.png')
    plt.savefig(save_path)
    plt.show()
    print(f"📊 Визуализация распределения оптимальных методов сохранена в '{save_path}'.")


def visualize_best_method_overall(best_method):
    """
    Визуализация лучшего метода по всем изображениям.
    """
    # Удаляем эмодзи из текста
    clean_best_method = best_method.replace('🏆 ', '') if '🏆 ' in best_method else best_method

    plt.figure(figsize=(6, 6))
    plt.text(0.5, 0.5, clean_best_method, fontsize=20, ha='center', va='center')
    plt.title('Лучший метод по всем изображениям')
    plt.axis('off')

    save_path = os.path.join(VISUALIZATION_FOLDER, 'best_method_overall.png')
    plt.savefig(save_path)
    plt.show()
    print(f"📊 Визуализация лучшего метода сохранена в '{save_path}'.")


def visualize_num_edges_per_image(df_all):
    """
    Визуализация количества обнаруженных границ по каждому изображению для всех методов.
    """
    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=df_all, x='Image', y='Num Edges', hue='Method', alpha=0.7, palette='tab10')
    plt.title('Количество обнаруженных границ по каждому изображению для всех методов')
    plt.xlabel('Изображение')
    plt.ylabel('Количество обнаруженных границ')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    save_path = os.path.join(VISUALIZATION_FOLDER, 'num_edges_per_image.png')
    plt.savefig(save_path)
    plt.show()
    print(f"📊 Визуализация количества границ по изображениям сохранена в '{save_path}'.")


def visualize_best_parameters(df_all, df_optimal):
    """
    Визуализация параметров, приведших к наибольшему количеству границ.
    """
    df_best_params = pd.merge(df_all, df_optimal, on='Image', suffixes=('', '_Optimal'))
    df_best_params = df_best_params[df_best_params['Method'] == df_best_params['Method_Optimal']]

    df_canny = df_best_params[df_best_params['Method'] == 'Canny']
    if not df_canny.empty:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Gaussian Kernel', data=df_canny, palette='pastel')
        plt.title('Распределение размеров ядра Гаусса для оптимального метода Canny')
        plt.xlabel('Размер ядра Гаусса')
        plt.ylabel('Количество изображений')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_path = os.path.join(VISUALIZATION_FOLDER, 'optimal_gaussian_kernel_canny.png')
        plt.savefig(save_path)
        plt.show()
        print(f"📊 Визуализация размеров ядра Гаусса для Canny сохранена в '{save_path}'.")

        # Визуализация пороговых значений для метода Canny
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df_canny,
                        x='Lower Threshold', y='Upper Threshold', hue='Gaussian Kernel', palette='viridis', s=100)
        plt.title('Пороговые значения для оптимального метода Canny')
        plt.xlabel('Нижний порог')
        plt.ylabel('Верхний порог')
        plt.legend(title='Размер ядра Гаусса')
        plt.tight_layout()
        save_path = os.path.join(VISUALIZATION_FOLDER, 'optimal_canny_thresholds.png')
        plt.savefig(save_path)
        plt.show()
        print(f"📊 Визуализация пороговых значений для Canny сохранена в '{save_path}'.")
    else:
        print("⚠️ Нет оптимальных параметров для метода Canny для визуализации.")


def main():
    df_all, df_optimal, best_method = read_data()

    visualize_num_edges_per_method(df_all)

    visualize_optimal_methods(df_optimal)

    visualize_best_method_overall(best_method)

    visualize_num_edges_per_image(df_all)

    visualize_best_parameters(df_all, df_optimal)


if __name__ == "__main__":
    main()
