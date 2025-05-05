import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(original, generated):
    mse = np.mean((original - generated) ** 2)
    if mse == 0:
        return float('inf')  # 两幅图像完全相同
    max_pixel = 255.0  # 对于8位图像
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_ssim(original, generated):
    # 指定 data_range
    return ssim(original, generated, data_range=255, multichannel=False)  # 对于灰度图像，multichannel=False

def main(original_image_path, generated_image_path):
    # 检查文件是否存在
    if not os.path.exists(original_image_path):
        print(f"文件不存在：{original_image_path}")
        return
    if not os.path.exists(generated_image_path):
        print(f"文件不存在：{generated_image_path}")
        return

    # 读取图像
    original = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取
    generated = cv2.imread(generated_image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取

    if original is None:
        print(f"无法读取原始图像：{original_image_path}")
        return

    if generated is None:
        print(f"无法读取生成图像：{generated_image_path}")
        return

    # 调整图像尺寸为224x224
    original = cv2.resize(original, (224, 224))
    generated = cv2.resize(generated, (224, 224))

    # 确保图像都是同样的尺寸
    if original.shape != generated.shape:
        print("图像尺寸不匹配！")
        return

    # 转换为浮点数
    original = original.astype(np.float32)
    generated = generated.astype(np.float32)

    # 计算PSNR和SSIM
    psnr_value = calculate_psnr(original, generated)
    ssim_value = calculate_ssim(original, generated)

    # 输出结果
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")

if __name__ == "__main__":
    # 替换为你的图像路径
    original_image_path = r'D:/target_images/Flir3/image_3.jpg'
    generated_image_path = r'D:/target_images/Flir3/image_2.jpg'

    main(original_image_path, generated_image_path)