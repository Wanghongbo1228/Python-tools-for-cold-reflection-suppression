import os
import torch
import cv2
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from unet_model import unet

# 定义数据集类
class GrayscaleDataset:
    def __init__(self, input_images, transform=None):
        self.input_images = input_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = Image.open(self.input_images[idx]).convert('L')  # 转换为灰度图像

        if self.transform:
            input_image = self.transform(input_image)

        return input_image

# 测试函数
def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # 设置模型为评估模式

    with torch.no_grad():
        for idx, inputs in enumerate(test_loader):
            inputs = inputs.to(device)  # 确保输入在正确的设备上
            outputs = model(inputs)

            # 可视化结果
            visualize_results(inputs[0], outputs[0], idx)

# 可视化结果
def visualize_results(input_image, output_image, idx):
    input_image = input_image.squeeze(0).detach().cpu().numpy()  # 移除batch维度
    output_image = output_image.squeeze(0).detach().cpu().numpy()  # 移除batch维度

    # 确保输出图像的形状是 (224, 224)
    if output_image.shape[0] == 1:  # 如果是单通道
        output_image = output_image[0]  # 只取第一个通道

    os.makedirs('./test_results', exist_ok=True)
    cv2.imwrite(f'./test_results/input_image_{idx}.png', input_image * 255)
    cv2.imwrite(f'./test_results/output_image_{idx}.png', output_image * 255)

# 主程序
if __name__ == "__main__":
    test_input_dir = 'D:/test_images'  # 测试图像路径
    test_images = [os.path.join(test_input_dir, f) for f in os.listdir(test_input_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset = GrayscaleDataset(test_images, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 实例化模型
    model = unet(input_channels=1, output_channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 确保模型在正确的设备上

    # 加载模型权重
    model.load_state_dict(torch.load("D:/PaperCode/unet/new_model.pt"))

    # 测试模型
    test_model(model, test_loader)