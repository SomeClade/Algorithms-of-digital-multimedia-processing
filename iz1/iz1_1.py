import cv2
import struct
import sys

from methods.CamShift import CamShift
from methods.KCFTracker import KCFTracker
from methods.MeanShift import MeanShiftHands
from methods.MedianFlow import MedianFlowTracker
from methods.CSRTTracker import CSRTTracker
from methods.TemplateMatchTracker import TemplateMatchTracker


# Функция для трекинга объектов
def track_object(video_path, tracker_type, **kwargs):
    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        sys.exit(1)

    # Получаем информацию о видео
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Выводим информацию о видео
    try:
        codec_name = struct.pack('I', fourcc).decode('utf-8')
    except:
        codec_name = "Неизвестен"
    print(f"Кодек видео: {codec_name}")
    print(f"Частота кадров видео: {fps}")
    print(f"Длительность видео: {duration}s")
    print(f"Разрешение видео: {width}x{height}")

    # Инициализация трекера в зависимости от выбранного типа
    if tracker_type == 'CamShift':
        tracker = CamShift()
    elif tracker_type == 'KCF':
        tracker = KCFTracker()
    elif tracker_type == 'MedianFlow':
        tracker = MedianFlowTracker()
    elif tracker_type == "MeanShiftHands":
        tracker = MeanShiftHands()
    elif tracker_type == "CSRT":
        tracker = CSRTTracker()
    elif tracker_type == "TemplateMatch":
        tracker = TemplateMatchTracker()
    else:
        print("Неизвестный тип трекера")
        return

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tracker.setUpWriter(fourcc, fps, (width, height))

    # Запускаем процесс трекинга
    tracker.process(cap, **kwargs)


if __name__ == "__main__":
    video_path = 'video4.mp4'
    #разные трекеры: 'TemplateMatch' 'CamShift', 'KCF',  упс 'MedianFlow', 'MeanShiftHands', 'CSRT'

    track_object(video_path, 'TemplateMatch')
