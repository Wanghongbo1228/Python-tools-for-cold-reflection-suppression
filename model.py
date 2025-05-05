import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np

# 自定义权重初始化
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# 定义数据集类
class GrayscaleDataset(Dataset):
    def __init__(self, input_images, target_images, transform=None):
        self.input_images = input_images
        self.target_images = target_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        try:
            input_image = Image.open(self.input_images[idx]).convert('L')
            target_image = Image.open(self.target_images[idx]).convert('L')

            if self.transform:
                input_image = self.transform(input_image)
                target_image = self.transform(target_image)

            return input_image, target_image
        except Exception as e:
            print(f"Error loading image {self.input_images[idx]}: {e}")
            return None, None

class down(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batch'):
        super(down, self).__init__()
        norm_layer = nn.BatchNorm2d(out_ch) if norm == 'batch' else nn.InstanceNorm2d(out_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            norm_layer,
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class up(nn.Module):
    def __init__(self, in_ch, out_ch, concat=True, final=False, norm='batch'):
        super(up, self).__init__()
        self.concat = concat
        self.final = final
        norm_layer = nn.BatchNorm2d(out_ch) if norm == 'batch' else nn.InstanceNorm2d(out_ch)

        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        ]

        if not final:
            layers += [norm_layer, nn.ReLU(inplace=True)]
        else:
            layers += [nn.Tanh()]

        self.conv = nn.Sequential(*layers)

    def forward(self, x1, x2):
        if self.concat and x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2,
                                         diffY // 2, diffY - diffY // 2))
            x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class unet(nn.Module):
    def __init__(self, input_channels, output_channels, norm='batch'):
        super(unet, self).__init__()
        self.down1 = down(input_channels, 64, norm)
        self.down2 = down(64, 128, norm)
        self.down3 = down(128, 256, norm)
        self.down4 = down(256, 512, norm)
        self.down5 = down(512, 512, norm)

        self.up1 = up(512, 512, concat=False, norm=norm)
        self.up2 = up(1024, 256, norm=norm)
        self.up3 = up(512, 128, norm=norm)
        self.up4 = up(256, 64, norm=norm)
        self.up5 = up(128, output_channels, final=True, norm=norm)

    def forward(self, x):
        # 下采样
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # 上采样
        u1 = self.up1(d5, None)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return u5

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()

    def vgg_preprocess(self, x):
        # 输入x范围[-1,1]，转为ImageNet归一化
        x = (x + 1) / 2
        x = x.repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std

    def forward(self, x, y):
        x = self.vgg_preprocess(x)
        y = self.vgg_preprocess(y)
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return nn.functional.mse_loss(x_features, y_features)

def train_val_data_process(input_dir, target_dir, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    input_images = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    target_images = sorted(
        [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    # 确保输入输出配对
    assert len(input_images) == len(target_images), "输入和目标文件数量不匹配"
    dataset = GrayscaleDataset(input_images, target_images, transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def visualize_results(input_tensor, target_tensor, output_tensor, epoch, prefix="train"):
    os.makedirs('./results', exist_ok=True)

    def tensor_to_image(tensor):
        # 输入tensor范围[-1,1]，转为0-255 numpy数组
        img = tensor.squeeze().detach().cpu().numpy()  # 使用detach()
        img = (img * 0.5 + 0.5) * 255
        return img.astype(np.uint8)

    input_img = tensor_to_image(input_tensor)
    target_img = tensor_to_image(target_tensor)
    output_img = tensor_to_image(output_tensor)

    # 拼接对比图像
    comparison = np.hstack([input_img, output_img, target_img])
    cv2.imwrite(f'./results/{prefix}_epoch{epoch:03d}.jpg', comparison)

def train_model_process(model, train_loader, val_loader, num_epochs=100, lr=1e-4, patience=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.apply(init_weights)
    model.to(device)

    criterion_l1 = nn.L1Loss()
    criterion_perceptual = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if inputs is None or targets is None:  # 检查None
                continue
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # 损失计算
            l1_loss = criterion_l1(outputs, targets)
            percep_loss = criterion_perceptual(outputs, targets)
            loss = l1_loss + 0.1 * percep_loss

            # 反向传播
            if not torch.isnan(loss).any():  # 处理NaN
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # 可视化
            if batch_idx == 0:
                visualize_results(inputs[0], targets[0], outputs[0], epoch, "train")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                if inputs is None or targets is None:  # 检查None
                    continue
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                l1_loss = criterion_l1(outputs, targets)
                percep_loss = criterion_perceptual(outputs, targets)
                loss = l1_loss + 0.1 * percep_loss
                val_loss += loss.item()

                if batch_idx == 0:
                    visualize_results(inputs[0], targets[0], outputs[0], epoch, "val")

        # 统计指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 早停和保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    torch.save(model.state_dict(), "final_model.pth")
    return model

if __name__ == "__main__":
    # 数据路径配置
    input_dir = 'D:/infrared_images'
    target_dir = 'D:/target_images/Flir1'

    # 数据加载
    train_loader, val_loader = train_val_data_process(input_dir, target_dir, batch_size=8)

    # 模型初始化
    model = unet(input_channels=1, output_channels=1, norm='batch')

    # 训练模型
    trained_model = train_model_process(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        lr=1e-4,
        patience=7
    )