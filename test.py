import cv2

extensions = ['jpg', 'png', 'bmp']

read_flags = [cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED]

window_flags = [cv2.IMREAD_UNCHANGED, cv2.IMREAD_UNCHANGED]

for ext in extensions:
    filename = f'img1.{ext}' #формат файла
    for read_flag in read_flags:
        img = cv2.imread(filename, read_flag)
        if img is None:
            print("image not found")
            continue
        for window_flag in window_flags:
            cv2.namedWindow('Image', window_flag)  # Создаем окно с флагом
            cv2.imshow('Image', img)
            print (f"Открыто изображение {filename} с флагом {read_flag} и флагом окна {window_flag}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()