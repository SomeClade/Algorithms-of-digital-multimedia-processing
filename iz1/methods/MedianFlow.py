import cv2

class MedianFlowTracker:
    def __init__(self):
        self.writer = None

    def setUpWriter(self, fourcc, fps, framesize):
        self.writer = cv2.VideoWriter("../resultMedianFlowTracker.mp4", fourcc, fps, framesize)

    def process(self, videoCap: cv2.VideoCapture, **kwargs):
        ret, frame = videoCap.read()
        if not ret:
            print("Не удалось прочитать видео")
            return

        bbox = cv2.selectROI("Tracking", frame, False)

        tracker = cv2.legacy.TrackerMedianFlow.create()
        tracker.init(frame, bbox)

        while True:
            ret, frame = videoCap.read()
            if not ret:
                break

            success, bbox = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Tracking", frame)
            self.writer.write(frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        self.writer.release()
        videoCap.release()
        cv2.destroyAllWindows()
