import numpy as np
import colorsys


def hue_shift_image_by_mean_brightness(image_array):
    # 确保输入的图像是一个 3D 数组，并且通道数是 3
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Input image array must have shape (H, W, 3).")

    # 创建一个新的数组来存储色相偏移后的图像
    shifted_image = np.zeros_like(image_array)

    # 将图像的 RGB 转换为亮度数组，计算每个像素的亮度
    brightness_array = 0.299 * image_array[:, :, 0] + 0.587 * image_array[:, :, 1] + 0.114 * image_array[:, :, 2]
    # brightness_array = 0.333 * image_array[:, :, 0] + 0.333 * image_array[:, :, 1] + 0.333 * image_array[:, :, 2]
    mean_brightness = np.median(brightness_array) / 255  # 归一化到 [0, 1]

    # 获取图像的高和宽
    h, w, _ = image_array.shape

    # 遍历图像的每个像素
    for i in range(h):
        for j in range(w):
            # 获取当前像素的 RGB 值，并归一化到 [0, 1]
            r, g, b = image_array[i, j] / 255.0

            # 转换到 HSV 空间
            h, s, v = colorsys.rgb_to_hsv(r, g, b)

            # 计算亮度差异，并根据差异方向调整色相
            if v > mean_brightness:
                # 亮度高于均值，逆时针偏移（左移）
                hue_shift = 0.1 * (v - mean_brightness) - 0.2
                # v = (v * 1.2) % 1.0
                # print(f'up: {hue_shift}')
            else:
                # 亮度低于均值，顺时针偏移（右移）
                hue_shift = 0.2 * (mean_brightness - v) + 0.2
                # v = (v * 0.8) % 1.0
                # print(f'down: {hue_shift}')

            # 应用色相偏移并限制到 [0, 1]
            h = (h + hue_shift) % 1.0
            # print(f'hue: {h}, shift: {hue_shift}')

            # 转换回 RGB 空间
            r, g, b = colorsys.hsv_to_rgb(h, s, v)

            # 存储到新的图像数组中，并缩放回 [0, 255]
            shifted_image[i, j] = (int(r * 255), int(g * 255), int(b * 255))

    return shifted_image


# 示例：加载并处理一个示例图像（需要使用 PIL 加载图像）
from PIL import Image

# 加载图像并转换为 ndarray
input_image = Image.open("c:/Users/Administrator/Downloads/IM_截圖_2024-10-29-16-39-16.png").convert("RGB")
image_array = np.array(input_image)

# 应用色相偏移
shifted_image_array = hue_shift_image_by_mean_brightness(image_array)

# 将结果转换为 PIL 图像并显示或保存
container = np.zeros((image_array.shape[0], image_array.shape[1] * 2, 3))
container[:, :image_array.shape[1], :] = image_array
container[:, image_array.shape[1]:, :] = shifted_image_array
shifted_image = Image.fromarray(container.astype("uint8"))
shifted_image.show()
# 或者保存：shifted_image.save("shifted_image.jpg")
