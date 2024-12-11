import cv2

class TemplateMatchTracker:
    """
    Трекер, основанный на сопоставлении шаблона.
    ------------------------------------------------------
    Как работает:
    1. На первом кадре пользователь выбирает ROI.
    2. Шаблон (ROI) запоминается.
    3. На каждом новом кадре мы ищем наиболее похожую зону с помощью cv2.matchTemplate.
    4. Определяем позицию bounding box на основе найденной позиции и рисуем его.

    Недостатки:
    - Неустойчив к изменению масштаба, угла или формы объекта.
    - Может быстро терять объект при изменении условий съемки.
    """
    def __init__(self):
        self.writer = None
        self.template = None  # Сам шаблон
        self.template_w = None
        self.template_h = None
        self.method = cv2.TM_CCOEFF_NORMED  # Метод сопоставления
        self.bbox = None

    def setUpWriter(self, fourcc, fps, framesize):
        # Инициализация VideoWriter
        self.writer = cv2.VideoWriter("../resultTemplateMatchTracker.mp4", fourcc, fps, framesize)

    def process(self, cap: cv2.VideoCapture, **kwargs):
        # Читаем первый кадр
        ret, frame = cap.read()
        if not ret:
            print("Не удалось прочитать видео")
            return

        # Выбор ROI для трекинга
        self.bbox = cv2.selectROI("Select ROI (TemplateMatch)", frame, fromCenter=False, showCrosshair=True)
        x, y, w, h = self.bbox

        # Вырезаем шаблон
        self.template = frame[y:y+h, x:x+w].copy()
        self.template_h, self.template_w = self.template.shape[:2]

        # Основной цикл
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Выполняем сопоставление шаблона
            res = cv2.matchTemplate(frame, self.template, self.method)

            # Находим позицию максимального совпадения
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Координаты совпадения: max_loc - верхний левый угол найденной области
            top_left = max_loc
            bottom_right = (top_left[0] + self.template_w, top_left[1] + self.template_h)

            # Рисуем прямоугольник
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            # Показываем результат
            cv2.imshow("Template Matching Tracking", frame)
            self.writer.write(frame)

            # Выход по нажатию 'q'
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        self.writer.release()
        cap.release()
        cv2.destroyAllWindows()
