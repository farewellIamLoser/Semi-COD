import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
# 假设你的矩阵名为cuda_tensor，形状为[3, 224, 224]
# 将CUDA张量移动到CPU内


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def main(image):
    a = image[0]
    a = a.view(-1, 224, 224)
    cpu_tensor = a.cpu()
    # 转换为灰度图像
    gray_tensor = torch.mean(cpu_tensor, dim=0)  # 取三个通道的平均值得到灰度图像的单通道
    # 创建灰度图像显示的transforms
    transform = transforms.ToPILImage(mode='L')
    # 转换为灰度图像
    gray_image = transform(gray_tensor)
    # 显示灰度图像
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.show()

def show(image):
    cpu_tensor = image.cpu()
    # 转换为灰度图像
    gray_tensor = torch.mean(cpu_tensor, dim=0)  # 取三个通道的平均值得到灰度图像的单通道
    # 创建灰度图像显示的transforms
    transform = transforms.ToPILImage(mode='L')
    # 转换为灰度图像
    gray_image = transform(gray_tensor)
    # 显示灰度图像
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.show()

def showRGB(image):
    cpu_tensor = image.cpu()
    # 转换为RGB图像
  # 将通道维度调整到最后一个维度
    # 创建RGB图像显示的transforms
    transform = transforms.ToPILImage(mode='RGB')
    # 转换为RGB图像
    rgb_image = transform(cpu_tensor)
    # 显示RGB图像
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.show()

def show_image(image, title=''):
    # image is [H, W, 3]
    image = torch.einsum('nchw->nhwc', image).detach().cpu()
    image = image[0]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()
    return


def show_image_gray(image, title=''):
    # image is [H, W, 3]
    image = torch.einsum('nchw->nhwc', image).detach().cpu()
    image = image[0]
    assert image.shape[2] == 3

    # 转换为灰度图
    image_gray = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2GRAY)

    # 显示灰度图
    plt.imshow(image_gray, cmap='gray')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()
    return

def show_image_Image2(image, title=''):
    # image is [H, W, 3]
    image = torch.einsum('nchw->nhwc', image).detach().cpu()
    image = image[0]
    assert image.shape[2] == 3

    # 转换为PIL图像
    image_pil = Image.fromarray(np.uint8(image))

    # 转换为灰度图
    image_gray = image_pil.convert('L')

    # 显示灰度图
    plt.imshow(image_gray, cmap='gray')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()
    return

def show_binear_gray(image):
    import matplotlib.pyplot as plt
    # 将图像数据从(1, 224, 224)转换为(224, 224)
    image_data = image[0].permute(1, 2, 0).detach().cpu().numpy()

    # 展示灰度图
    plt.imshow(image_data, cmap='gray')
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    return