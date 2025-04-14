import cv2
import numpy as np
import os
# import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import shutil


def shift_matrix(matrix, shift_direction, stride=1):
    """
    根据给定的平移方向和步幅对矩阵进行上下左右平移
    :param matrix: 输入矩阵，形状为 (n, m)
    :param shift_direction: 平移方向，0~3 分别代表不同的平移方式：0表示向上，1表示向左，2表示向下，3表示向右
    :param stride: 平移的步幅
    :return: 平移后的矩阵
    """
    n, m = matrix.shape
    shifted_matrix = np.zeros_like(matrix)

    # 计算平移的行和列偏移量
    if shift_direction == 0:  # 向上平移
        row_offset = -stride
        col_offset = 0
    elif shift_direction == 1:  # 向左平移
        row_offset = 0
        col_offset = -stride
    elif shift_direction == 2:  # 向下平移
        row_offset = stride
        col_offset = 0
    elif shift_direction == 3:  # 向右平移
        row_offset = 0
        col_offset = stride
    else:
        row_offset = 0
        col_offset = 0

    # 切片操作进行平移
    shifted_matrix[max(0, row_offset):min(n, n + row_offset), max(0, col_offset):min(m, m + col_offset)] = \
        matrix[max(0, -row_offset):min(n, n - row_offset), max(0, -col_offset):min(m, m - col_offset)]

    return shifted_matrix


def calculate_gray_concentration(image_path):
    # 读取灰度图像
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 计算灰度直方图
    shift_img = shift_matrix(img_gray, shift_direction=1, stride=1)
    Grad_img = np.abs(img_gray - shift_img)
    Grad_img[Grad_img == 255] = 0
    mean = np.mean(Grad_img)
    mean = np.floor(mean)
    return mean


def plot_bar_from_list(data_list):
    # 使用 Counter 统计列表中每个元素的出现次数
    counter = Counter(data_list)

    # 获取元素和其对应的计数
    elements = list(counter.keys())
    counts = list(counter.values())

    # 绘制柱状图
    # plt.figure(figsize=(10, 6))  # 设置图的大小
    # plt.bar(elements, counts, color='skyblue')
    #
    # # 添加标题和标签
    # plt.title('Bar Chart of Duplicate Values')
    # plt.xlabel('Values')
    # plt.ylabel('Count')
    #
    # # 显示图形
    # plt.show()


def is_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # 检查文件是否是有效的图片
        return True
    except (IOError, SyntaxError) as e:
        return False


def delete_all_files_in_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 遍历并删除所有文件
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                print(f"Skipping non-file item: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


def count_images_in_folder(folder_path):
    image_count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if is_image(file_path):
                image_count += 1
    return image_count


def samimage_select(i):
    folder_path = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage3/SAM'  # 替换为你的图片文件夹路径
    temp_folder = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage3/TempScribble'
    dest_folder = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage3/Scribble'
    label_folder = "/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage1.txt"
    temp_count = count_images_in_folder(temp_folder)
    with open(label_folder, 'r') as file:
        label_names = {line.strip() for line in file}
    if temp_count == 0:
        print('first time')
    else:
        delete_all_files_in_folder(temp_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            evaluation = calculate_gray_concentration(image_path)

            if os.path.splitext(filename)[0] in label_names:
                print(filename)
                image_path = image_path.replace('SAM', 'GT')
                shutil.copy(image_path, os.path.join(temp_folder, filename))
                continue

            if evaluation < i:
                # 复制图片到目标文件夹
                shutil.copy(image_path, os.path.join(temp_folder, filename))
            else:
                image_path = image_path.replace('SAM', 'Prompt')
                shutil.copy(image_path, os.path.join(temp_folder, filename))

    temp_count = count_images_in_folder(temp_folder)
    dest_count = count_images_in_folder(dest_folder)

    if dest_count == 0:
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(folder_path, filename)
                evaluation = calculate_gray_concentration(image_path)

                if os.path.splitext(filename)[0] in label_names:
                    image_path = image_path.replace('SAM', 'GT')
                    shutil.copy(image_path, os.path.join(temp_folder, filename))
                    continue

                if evaluation < i:
                    # 复制图片到目标文件夹
                    shutil.copy(image_path, os.path.join(dest_folder, filename))
                else:
                    image_path = image_path.replace('SAM', 'Prompt')
                    shutil.copy(image_path, os.path.join(dest_folder, filename))
        # 获取文件夹中所有图片文件的文件名（不包括扩展名）
        image_files = [os.path.splitext(f)[0] for f in os.listdir(dest_folder) if
                       f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

        # 将文件名写入一个文本文件
        output_file = "/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage3.txt"
        with open(output_file, 'w') as file:
            for image_file in image_files:
                file.write(image_file + '\n')

        print(f"Image file names (without extensions) saved to {output_file}")
        return i
    elif temp_count <= dest_count:
        delete_all_files_in_folder(dest_folder)
        i += 1
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(folder_path, filename)
                evaluation = calculate_gray_concentration(image_path)

                if os.path.splitext(filename)[0] in label_names:
                    image_path = image_path.replace('SAM', 'GT')
                    shutil.copy(image_path, os.path.join(temp_folder, filename))
                    continue

                if evaluation < i:
                    # 复制图片到目标文件夹
                    shutil.copy(image_path, os.path.join(dest_folder, filename))
                else:
                    image_path = image_path.replace('SAM', 'Prompt')
                    shutil.copy(image_path, os.path.join(dest_folder, filename))
        # 获取文件夹中所有图片文件的文件名（不包括扩展名）
        image_files = [os.path.splitext(f)[0] for f in os.listdir(dest_folder) if
                       f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

        # 将文件名写入一个文本文件
        output_file = "/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage3.txt"
        with open(output_file, 'w') as file:
            for image_file in image_files:
                file.write(image_file + '\n')

        print(f"Image file names (without extensions) saved to {output_file}")
        return i
    elif temp_count > dest_count:
        delete_all_files_in_folder(dest_folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(folder_path, filename)
                evaluation = calculate_gray_concentration(image_path)
                if filename in label_names:
                    image_path = image_path.replace('SAM', 'GT')
                    shutil.copy(image_path, os.path.join(temp_folder, filename))
                    continue

                if evaluation < i:
                    # 复制图片到目标文件夹
                    shutil.copy(image_path, os.path.join(dest_folder, filename))
                else:
                    image_path = image_path.replace('SAM', 'Prompt')
                    shutil.copy(image_path, os.path.join(dest_folder, filename))
        # 获取文件夹中所有图片文件的文件名（不包括扩展名）
        image_files = [os.path.splitext(f)[0] for f in os.listdir(dest_folder) if
                       f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

        # 将文件名写入一个文本文件
        output_file = "/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage3.txt"
        with open(output_file, 'w') as file:
            for image_file in image_files:
                file.write(image_file + '\n')

        print(f"Image file names (without extensions) saved to {output_file}")
        return i

samimage_select(3)