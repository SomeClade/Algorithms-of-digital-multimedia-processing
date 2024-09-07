import cv2

image = cv2.imread('images/img1.jpg')

if image is None:
    print("Ошибка открытия изображения")
    exit()

# меняем цвета в палитру hsv
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # BGR != RGB

# Исходное
resizeOriginalImage = cv2.resize(image, (400,600))
cv2.imshow('Original Image', resizeOriginalImage)

# HSV
resizeHSVImage = cv2.resize(hsv_image, (400,600))
cv2.imshow('HSV Image', resizeHSVImage)

cv2.waitKey(0)

cv2.destroyAllWindows()
