const Jimp = require('jimp').default; // For newer Jimp versions


async function applyGaussianBlur(imagePath, outputImagePath, radius) {
    try {
        const image = await Jimp.read(imagePath); // Читаем изображение
        image.gaussian(radius); // Применяем размытие Гаусса
        await image.writeAsync(outputImagePath); // Сохраняем результат
        console.log(`Gaussian Blur applied. Result saved at ${outputImagePath}`);
    } catch (error) {
        console.error('Error applying Gaussian blur:', error);
    }
}

applyGaussianBlur('images/img1.jpg', 'blurred_image_jimp.jpg', 5);

// не робит на ноде, открыть webstorm