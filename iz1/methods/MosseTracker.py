import cv2 as cv
import numpy as np

eps = 1e-5


def divSpec(A, B):
    # Делим спектры комплексных матриц A и B (два канала: действительная и мнимая часть)
    Ar, Ai = A[..., 0], A[..., 1]
    Br, Bi = B[..., 0], B[..., 1]
    C = (Ar + 1j * Ai) / (Br + 1j * Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C


def my_dft(image):
    """
    image - 2D numpy массив действительных чисел.
    Возвращает комплексный спектр в виде массива complex.
    Пояснения:
    DFT по двум измерениям:
      F(u,v) = Σ_x Σ_y f(x,y)*exp(-2πi*(u*x/M + v*y/N))
    где M,N - размеры изображения.
    """
    M, N = image.shape
    # Создаем сетки индексов
    x = np.arange(M)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Преобразование по двум осям
    # Используем формулу DFT:
    # F(u,v) = Σ_x Σ_y f(x,y)*exp(-2πi*(x*u/M + y*v/N))
    # u,v пробегают по всем частотам
    F = np.zeros((M, N), dtype=complex)
    for u in range(M):
        for v in range(N):
            F[u, v] = np.sum(image * np.exp(-2j * np.pi * ((u * X / M) + (v * Y / N))))
    return F


class MOSSE:
    def __init__(self, frame, rect):
        x1, y1, x2, y2 = rect
        w = cv.getOptimalDFTSize(x2 - x1)
        h = cv.getOptimalDFTSize(y2 - y1)
        x1, y1 = (x1 + x2 - w) // 2, (y1 + y2 - h) // 2
        self.pos = x, y = x1 + 0.5 * (w - 1), y1 + 0.5 * (h - 1)
        self.size = w, h
        img = cv.getRectSubPix(frame, (w, h), (x, y))

        self.win = cv.createHanningWindow((w, h), cv.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h // 2, w // 2] = 1
        g = cv.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv.dft(g, flags=cv.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)

        test_img = self.preprocess(img)
        # my_dft_res = my_dft(test_img)  # Можно раскомментировать для демонстрации

        for _i in range(16):  # Уменьшим количество итераций для скорости
            a = self.preprocess(self.rnd_warp(img))
            A = cv.dft(a, flags=cv.DFT_COMPLEX_OUTPUT)
            self.H1 += cv.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2 += cv.mulSpectrums(A, A, 0, conjB=True)
        self.update_kernel()
        self.update(frame)

    def rnd_warp(self, a):
        h, w = a.shape[:2]
        T = np.zeros((2, 3))
        coef = 0.2
        ang = (np.random.rand() - 0.5) * coef
        c, s = np.cos(ang), np.sin(ang)
        T[:2, :2] = [[c, -s], [s, c]]
        T[:2, :2] += (np.random.rand(2, 2) - 0.5) * coef
        cx, cy = w / 2, h / 2
        T[:, 2] = [cx, cy] - np.dot(T[:2, :2], [cx, cy])
        return cv.warpAffine(a, T, (w, h), borderMode=cv.BORDER_REFLECT)

    def update_kernel(self):
        self.H = divSpec(self.H1, self.H2)
        self.H[..., 1] *= -1

    def preprocess(self, img):
        img = np.log(np.float32(img) + 1.0)
        img = (img - img.mean()) / (img.std() + eps)
        return img * self.win

    def correlate(self, img):
        C = cv.mulSpectrums(cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv.idft(C, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
        h, w = resp.shape
        _, mval, _, (mx, my) = cv.minMaxLoc(resp)
        side_resp = resp.copy()
        cv.rectangle(side_resp, (mx - 5, my - 5), (mx + 5, my + 5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval - smean) / (sstd + eps)
        return resp, (mx - w // 2, my - h // 2), psr

    def update(self, frame, rate=0.125):
        (x, y), (w, h) = self.pos, self.size
        self.last_img = img = cv.getRectSubPix(frame, (w, h), (x, y))
        img = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        self.good = self.psr > 8.0
        if not self.good:
            return
        self.pos = x + dx, y + dy
        self.last_img = img2 = cv.getRectSubPix(frame, (w, h), self.pos)
        img2 = self.preprocess(img2)
        A = cv.dft(img2, flags=cv.DFT_COMPLEX_OUTPUT)
        H1 = cv.mulSpectrums(self.G, A, 0, conjB=True)
        H2 = cv.mulSpectrums(A, A, 0, conjB=True)
        self.H1 = self.H1 * (1.0 - rate) + H1 * rate
        self.H2 = self.H2 * (1.0 - rate) + H2 * rate
        self.update_kernel()

    @property
    def state_vis(self):
        f = cv.idft(self.H, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
        h, w = f.shape
        f = np.roll(f, -h // 2, 0)
        f = np.roll(f, -w // 2, 1)
        kernel = np.uint8((f - f.min()) / f.ptp() * 255) if f.ptp() != 0 else f
        resp = self.last_resp
        resp_norm = resp / resp.max() if resp.max() != 0 else resp
        resp = np.uint8(np.clip(resp_norm, 0, 1) * 255)
        # Выводим состояние рядом: последнее изображение, ядро и респонс
        # Приведем все к одному размеру по высоте
        vis = np.hstack([cv.cvtColor(np.uint8(np.clip(self.last_img, 0, 255)), cv.COLOR_GRAY2BGR),
                         cv.cvtColor(kernel, cv.COLOR_GRAY2BGR),
                         cv.cvtColor(resp, cv.COLOR_GRAY2BGR)])
        return vis

    def draw_state(self, vis):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x - 0.5 * w), int(y - 0.5 * h), int(x + 0.5 * w), int(y + 0.5 * h)
        color = (0, 0, 255) if self.good else (0, 255, 255)
        cv.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv.putText(vis, f'PSR: {self.psr:.2f}', (x1, y2 + 15), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))


class MosseTracker:
    """
    При запуске:
    - На первом кадре выбираем ROI
    - Инициализируем MOSSE
    - В цикле читаем кадры, обновляем MOSSE-трекер, выводим результат.
    """

    def __init__(self):
        self.writer = None
        self.mosse = None

    def setUpWriter(self, fourcc, fps, framesize):
        self.writer = cv.VideoWriter("../result.mp4", fourcc, fps, framesize)

    def process(self, cap: cv.VideoCapture, **kwargs):
        ret, frame = cap.read()
        if not ret:
            print("Не удалось прочитать видео")
            return

        # Преобразуем в градации серого, MOSSE работает с одноканальным изображением
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Выбор ROI
        bbox = cv.selectROI("Select ROI for MOSSE", frame, fromCenter=False, showCrosshair=True)
        x, y, w, h = bbox
        x2 = x + w
        y2 = y + h

        # Инициализация MOSSE
        self.mosse = MOSSE(frame_gray, (x, y, x2, y2))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Обновляем MOSSE
            self.mosse.update(frame_gray)

            # Визуализация
            vis = frame.copy()
            self.mosse.draw_state(vis)
            cv.imshow("MOSSE Tracking", vis)
            if self.writer is not None:
                self.writer.write(vis)

            state_vis = self.mosse.state_vis
            cv.imshow("MOSSE State", state_vis)

            if cv.waitKey(30) & 0xFF == ord('q'):
                break

        if self.writer is not None:
            self.writer.release()
        cap.release()
        cv.destroyAllWindows()
