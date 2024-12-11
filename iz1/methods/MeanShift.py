import cv2
import numpy as np


class MeanShiftHands:
    # Класс для трекинга на основе MeanShift (руки или другие объекты)
    def __init__(self):
        self.writer = None

    def setUpWriter(self, fourcc, fps, framesize):
        self.writer = cv2.VideoWriter("../resultMeanShiftHands.mp4", fourcc, fps, framesize)

    def createMask(self, frame, low, top):
        # Создает маску по цвету из нескольких диапазонов
        resultMask = np.zeros_like(frame[:, :, 0])
        for i in range(len(low)):
            mask = cv2.inRange(frame, low[i], top[i])
            resultMask = cv2.bitwise_or(resultMask, mask)
        return resultMask

    def camshift_manual(self, frame, roi_hist, bbox, max_iter=10, epsilon=1):
        # Ручная реализация поиска положения объекта на основе гистограммы
        x, y, w, h = bbox
        for _ in range(max_iter):
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            back_proj = cv2.calcBackProject([hsv_frame], [0], roi_hist, [0, 180], 1)

            roi = back_proj[y:y + h, x:x + w]
            moments = cv2.moments(roi)
            if moments['m00'] == 0:
                break

            dx = int(moments['m10'] / moments['m00'])
            dy = int(moments['m01'] / moments['m00'])

            new_x = x + dx - w // 2
            new_y = y + dy - h // 2

            if abs(new_x - x) < epsilon and abs(new_y - y) < epsilon:
                break

            x, y = new_x, new_y

        return (x, y, w, h)

    def process(self, cap: cv2.VideoCapture, **kwargs):
        ret, frame = cap.read()
        if not ret:
            print("Не удалось захватить кадр.")
            cap.release()
            return

        bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        x, y, w, h = bbox

        roi = frame[y:y + h, x:x + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = None
        if 'bottom' in kwargs and 'top' in kwargs:
            mask = self.createMask(hsv_roi, kwargs['bottom'], kwargs['top'])
            if mask is not None and mask.size > 0:
                cv2.imshow('Mask', mask)

        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            bbox = self.camshift_manual(frame, roi_hist, bbox)
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Tracking", frame)
            self.writer.write(frame)

            if cv2.waitKey(30) & 0xFF == 27:  # ESC для выхода
                break

        self.writer.release()
        cap.release()
        cv2.destroyAllWindows()
