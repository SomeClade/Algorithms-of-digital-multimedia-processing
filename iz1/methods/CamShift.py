import cv2
import numpy as np


class CamShift:
    # Класс для трекинга с помощью CamShift
    def __init__(self):
        self.writer = None

    def setUpWriter(self, fourcc, fps, framesize):
        self.writer = cv2.VideoWriter("../result.mp4", fourcc, fps, framesize)

    def createMask(self, frame, low, top):
        # Создает маску по заданным нижним и верхним порогам
        resultMask = np.zeros_like(frame[:, :, 0])
        for i in range(len(low)):
            mask = cv2.inRange(frame, low[i], top[i])
            resultMask = cv2.bitwise_or(resultMask, mask)
        return resultMask

    def process(self, cap: cv2.VideoCapture, **kwargs):
        # Чтение первого кадра
        ret, frame = cap.read()
        if not ret:
            print("Не удалось прочитать видео")
            return

        # Выбор ROI для трекинга
        bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        (x, y, w, h) = bbox
        track_window = (x, y, w, h)

        # Инициализация гистограммы
        roi = frame[y:y + h, x:x + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = None
        if 'bottom' in kwargs and 'top' in kwargs:
            mask = self.createMask(hsv_roi, kwargs['bottom'], kwargs['top'])

        if mask is not None and mask.size > 0:
            cv2.imshow('Mask', mask)

        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Критерии остановки для CamShift
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if 'bottom' in kwargs and 'top' in kwargs:
                mask = self.createMask(hsv, kwargs['bottom'], kwargs['top'])
                if mask is not None and mask.size > 0:
                    cv2.imshow('Mask', mask)

            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            cv2.imshow('BackProj', dst)

            ret_camshift, track_window = cv2.CamShift(dst, track_window, term_crit)
            pts = cv2.boxPoints(ret_camshift)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            cv2.imshow('Result', img2)

            self.writer.write(img2)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        self.writer.release()
        cap.release()
        cv2.destroyAllWindows()
