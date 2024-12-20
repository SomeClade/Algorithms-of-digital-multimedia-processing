import cv2
import os
import numpy as np
import time
import threading

CARDS_PATH = 'cards'

ORB_FEATURES = 1000

ROUND_TIME = 10

orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

card_features = {}

algorithm_found = False
algorithm_time = None
common_symbol_coords = None  # Координаты общей картинки
current_pair = (None, None)  # Текущая пара карт
# нужно для корректной записи картинок
round_over = False
lock = threading.Lock()

# Глобальные переменные для второго окна пользователя
user_selected_points = []
user_round_over = False


def load_cards(path):
    cards = []
    for filename in os.listdir(path):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                cards.append((filename, img))
            else:
                print(f"Не удалось загрузить изображение: {img_path}")
    return cards


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def compute_features(cards):
    """
    Вычисляет ключевые точки и дескрипторы для каждой карты.
    """
    for name, img in cards:
        gray = preprocess_image(img)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is not None:
            card_features[name] = (keypoints, descriptors)
        else:
            print(f"Дескрипторы не найдены для карты: {name}")


def find_common_symbol(card1_name, card2_name):
    """
    Находит общую картинку между двумя картами.
    Возвращает координаты ключевых точек общей картинки или None.
    """
    global algorithm_found, algorithm_time, common_symbol_coords, round_over
    if card1_name not in card_features or card2_name not in card_features:
        print("Одна из карт не была загружена или обработана.")
        with lock:
            algorithm_found = False
            algorithm_time = 0
            common_symbol_coords = None
            round_over = True
        return

    kp1, des1 = card_features[card1_name]
    kp2, des2 = card_features[card2_name]

    if des1 is None or des2 is None:
        print("Дескрипторы не найдены для одной из карт.")
        with lock:
            algorithm_found = False
            algorithm_time = 0
            common_symbol_coords = None
            round_over = True
        return

    start_time = time.perf_counter()

    # Сопоставление дескрипторов
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    # Определение порога для хороших совпадений
    good_matches = [m for m in matches if m.distance < 60]

    if len(good_matches) == 0:
        print("Алгоритм не нашёл хороших совпадений.")
        end_time = time.perf_counter()
        processing_time = end_time - start_time

        with lock:
            algorithm_found = False
            algorithm_time = processing_time
            common_symbol_coords = None
            round_over = True

        print(f"Алгоритм не нашёл общую картинку. Время обработки: {processing_time:.2f} секунд.")
        return

    # Предполагаем, что первое хорошее совпадение - это общая картинка
    common_kp1 = kp1[good_matches[0].queryIdx]
    common_kp2 = kp2[good_matches[0].trainIdx]

    # Координаты общей картинки
    x1, y1 = common_kp1.pt
    x2, y2 = common_kp2.pt

    end_time = time.perf_counter()
    processing_time = end_time - start_time

    with lock:
        algorithm_found = True
        algorithm_time = processing_time
        common_symbol_coords = ((int(x1), int(y1)), (int(x2), int(y2)))
        round_over = True

    print(f"Алгоритм нашёл общую картинку за {algorithm_time:.2f} секунд.")
    print(f"Original Coordinates: {common_symbol_coords}")


def highlight_common_symbol(card1_img, card2_img, point1, point2, size=80, thickness=8):
    """
    Отмечает общую картинку на обеих картах толстыми красными прямоугольниками.
    Размер и толщина можно регулировать для большей видимости.
    """
    if point1:
        top_left1 = (point1[0] - size // 2, point1[1] - size // 2)
        bottom_right1 = (point1[0] + size // 2, point1[1] + size // 2)
        cv2.rectangle(card1_img, top_left1, bottom_right1, (0, 0, 255), thickness)
        cv2.rectangle(card1_img, top_left1, bottom_right1, (0, 0, 255), thickness // 2)

    if point2:
        top_left2 = (point2[0] - size // 2, point2[1] - size // 2)
        bottom_right2 = (point2[0] + size // 2, point2[1] + size // 2)
        cv2.rectangle(card2_img, top_left2, bottom_right2, (0, 0, 255), thickness)
        cv2.rectangle(card2_img, top_left2, bottom_right2, (0, 0, 255), thickness // 2)

    return card1_img, card2_img


def resize_image(img, max_size=600):
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        scaling_factor = max_size / float(max(height, width))
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return resized_img, scaling_factor
    return img, 1.0


def select_unique_pairs(cards):
    """
    Генерирует пары, где первое изображение сравнивается с каждым из остальных изображений.
    """
    if not cards:
        return []
    first_card = cards[0]
    return [(first_card, other_card) for other_card in cards[1:]]


def on_mouse_event_user(event, x, y, flags, param):
    global user_selected_points, user_round_over
    if event == cv2.EVENT_LBUTTONDOWN and not user_round_over:
        if len(user_selected_points) < 2:
            user_selected_points.append((x, y))
            print(f"Пользователь выбрал точку: ({x}, {y})")
            if len(user_selected_points) == 2:
                user_round_over = True
                print("Пользователь выделил два объекта.")


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_common_symbol(card_img, point, size, save_path):
    """
    Вырезает область вокруг заданной точки из изображения и сохраняет её.
    """
    x, y = point
    half_size = size // 2
    height, width = card_img.shape[:2]

    # Определение границ вырезки с учетом границ изображения
    top = max(y - half_size, 0)
    bottom = min(y + half_size, height)
    left = max(x - half_size, 0)
    right = min(x + half_size, width)

    print(f"Cropping region: top={top}, bottom={bottom}, left={left}, right={right} for save_path={save_path}")

    # Вырезание области
    cropped_img = card_img[top:bottom, left:right]

    if cropped_img.size == 0:
        print(f"Warning: Cropped image is empty for save_path={save_path}. Skipping save.")
    else:
        try:
            # Сохранение вырезанного изображения
            cv2.imwrite(save_path, cropped_img)
            print(f"Сохранён общий символ: {save_path}")
        except Exception as e:
            print(f"Ошибка при сохранении изображения {save_path}: {e}")


def play_round(card1, card2, scores, round_number):
    """
    Играет один раунд: отображает две карты, запускает алгоритм и ожидает действия пользователя.
    """
    global algorithm_found, algorithm_time, common_symbol_coords, round_over, user_selected_points, user_round_over
    algorithm_found = False
    algorithm_time = None
    common_symbol_coords = None
    round_over = False
    user_selected_points = []
    user_round_over = False

    algorithm_thread = threading.Thread(target=find_common_symbol, args=(card1[0], card2[0]))
    algorithm_thread.start()

    # Подготовка изображений с изменением размера и получением коэффициентов масштабирования
    card1_img, scale1 = resize_image(card1[1].copy(), max_size=600)
    card2_img, scale2 = resize_image(card2[1].copy(), max_size=600)

    combined_image = np.hstack((card1_img, card2_img))

    window_name_user = "user"
    cv2.namedWindow(window_name_user, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_user, 1200, 600)
    cv2.setMouseCallback(window_name_user, on_mouse_event_user, param=None)

    window_name_algo = "algorithm"
    cv2.namedWindow(window_name_algo, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_algo, 1200, 600)

    start_time = time.perf_counter()

    while True:
        with lock:
            if common_symbol_coords:
                scaled_coords = (
                    (int(common_symbol_coords[0][0] * scale1), int(common_symbol_coords[0][1] * scale1)),
                    (int(common_symbol_coords[1][0] * scale2), int(common_symbol_coords[1][1] * scale2))
                )
                card1_marked, card2_marked = highlight_common_symbol(
                    card1_img.copy(),
                    card2_img.copy(),
                    scaled_coords[0],
                    scaled_coords[1]
                )
                combined_marked = np.hstack((card1_marked, card2_marked))
                cv2.imshow(window_name_algo, combined_marked)
            else:
                cv2.imshow(window_name_algo, combined_image)

        display_user_img = combined_image.copy()
        for point in user_selected_points:
            cv2.circle(display_user_img, point, 20, (255, 0, 0), 4)  # Синие круги с большей толщиной
        cv2.imshow(window_name_user, display_user_img)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Нажатие Esc для выхода
            cv2.destroyAllWindows()
            exit()

        # Проверка завершения раунда
        with lock:
            if round_over:
                break
        if user_round_over:
            print("Пользователь завершил выделение.")
            break

    algorithm_thread.join()

    with lock:
        if common_symbol_coords:
            scaled_coords = (
                (int(common_symbol_coords[0][0] * scale1), int(common_symbol_coords[0][1] * scale1)),
                (int(common_symbol_coords[1][0] * scale2), int(common_symbol_coords[1][1] * scale2))
            )
            card1_marked, card2_marked = highlight_common_symbol(
                card1_img.copy(),
                card2_img.copy(),
                scaled_coords[0],
                scaled_coords[1]
            )
            combined_marked = np.hstack((card1_marked, card2_marked))
            cv2.imshow(window_name_algo, combined_marked)
            cv2.waitKey(1000)

            ensure_directory_exists('common_symbols')
            size = 80
            symbol1_path = os.path.join('common_symbols', f'round_{round_number}_card1.png')
            symbol2_path = os.path.join('common_symbols', f'round_{round_number}_card2.png')
            save_common_symbol(card1_img, scaled_coords[0], size, symbol1_path)
            save_common_symbol(card2_img, scaled_coords[1], size, symbol2_path)

    cv2.destroyWindow(window_name_algo)
    cv2.destroyWindow(window_name_user)

    with lock:
        if algorithm_found:
            if user_round_over:
                # Проверяем, совпадают ли выделенные пользователем точки с координатами алгоритма
                user_correct = False
                for user_point in user_selected_points:
                    for algo_point in scaled_coords:
                        dist = np.sqrt((user_point[0] - algo_point[0]) ** 2 + (user_point[1] - algo_point[1]) ** 2)
                        if dist < 60:  # Радиус допустимой ошибки
                            user_correct = True
                            break
                    if user_correct:
                        break

                if user_correct:
                    user_time = time.perf_counter() - start_time
                    if user_time < algorithm_time:
                        print(f"Пользователь нашёл общую картинку быстрее алгоритма! ({user_time:.2f} < {algorithm_time:.2f})")
                        scores['user'] += 1
                    else:
                        print(f"Алгоритм нашёл общую картинку быстрее пользователя! ({algorithm_time:.2f} < {user_time:.2f})")
                        scores['algorithm'] += 1
                else:
                    print("Пользователь не смог найти общую картинку.")
                    scores['algorithm'] += 1
            else:
                print("Пользователь не выделил точки. Алгоритм получает очко.")
                scores['algorithm'] += 1
        else:
            if user_round_over:
                user_correct = True
                if user_correct:
                    print("Пользователь нашёл общую картинку, алгоритм не смог.")
                    scores['user'] += 1
                else:
                    print("Пользователь не смог найти общую картинку.")
            else:
                print("Раунд завершился без успешного нахождения общей картинки.")

    print(f"Счёт: Пользователь {scores['user']} - Алгоритм {scores['algorithm']}\n")


def main():
    print("Загрузка карт...")
    cards = load_cards(CARDS_PATH)
    print(f"Загружено {len(cards)} карт.")

    if len(cards) < 2:
        print("Недостаточно карт для игры. Необходимо как минимум 2 карты.")
        return

    print("Вычисление ключевых точек и дескрипторов...")
    compute_features(cards)

    pairs = select_unique_pairs(cards)
    print(f"Всего пар для сравнения: {len(pairs)}")

    scores = {'user': 0, 'algorithm': 0}

    for idx, pair in enumerate(pairs):
        round_number = idx + 1
        print(f"Раунд {round_number} из {len(pairs)}: {pair[0][0]} vs {pair[1][0]}")
        play_round(pair[0], pair[1], scores, round_number)
        print(f"Счёт: Пользователь {scores['user']} - Алгоритм {scores['algorithm']}\n")

    print("Игра завершена!")
    print(f"Финальный счёт: Пользователь {scores['user']} - Алгоритм {scores['algorithm']}")
    if scores['user'] > scores['algorithm']:
        print("Поздравляем! Вы победили алгоритм!")
    elif scores['user'] < scores['algorithm']:
        print("Алгоритм победил! Попробуйте снова.")
    else:
        print("Ничья!")


if __name__ == "__main__":
    main()
