
# 文件复制阶段
import os
import shutil
import random
from collections import defaultdict
source_folder = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage1/GT'
destination_folder = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage1/Scribble'
destination_folder_1 = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage1/Scribble'
destination_folder_2 = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage2/Scribble'
def clear_files_in_folder(folder_path):
    # 列出文件夹中的所有内容
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # 如果是文件或链接，删除
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
clear_files_in_folder(destination_folder_1)
clear_files_in_folder(destination_folder_2)
# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 定义每个类别需要复制的图片数量
copies_per_category = 6  # 可以根据需要修改成任意整数

# 用于存储每个类别已复制的文件数量
copied_counts = {}

for filename in os.listdir(source_folder):
    # 获取文件的完整路径
    source_file_path = os.path.join(source_folder, filename)

    # 提取类别信息，例如 "4-Crocodile"
    parts = filename.split('-')
    if len(parts) >= 6:
        category = parts[4] + '-' + parts[5]

        # 初始化类别的复制计数
        if category not in copied_counts:
            copied_counts[category] = 0

        # 判断是否继续复制该类别的文件
        if copied_counts[category] < copies_per_category:
            # 构建目标文件的完整路径
            destination_file_path = os.path.join(destination_folder, filename)
            # 复制文件
            shutil.copy(source_file_path, destination_file_path)
            print(f'已复制文件: {filename} 到类别: {category}')
            # 更新复制计数
            copied_counts[category] += 1
        else:
            print(f'已达到类别: {category} 的复制数量上限，跳过: {filename}')
    else:
        print(f'文件名格式不正确，跳过: {filename}')

print('文件复制完成。')


source_folder = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage1/GT'

# 确保目标文件夹存在
os.makedirs(destination_folder_1, exist_ok=True)
os.makedirs(destination_folder_2, exist_ok=True)

# 定义每个类别需要复制的数量 # 这里设置每个类别需要复制的数量，可以根据需要修改

# 用于存储每个类别已复制的文件数量
copied_counts_1 = {}
copied_counts_2 = {}

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 获取文件的完整路径
    source_file_path = os.path.join(source_folder, filename)

    # 提取类别信息，例如 "4-Crocodile"
    parts = filename.split('-')
    if len(parts) >= 6:  # 确保文件名格式正确
        category = parts[4] + '-' + parts[5]

        # 初始化每个类别的复制计数
        if category not in copied_counts_1:
            copied_counts_1[category] = 0

        if category not in copied_counts_2:
            copied_counts_2[category] = 0

        # 判断是否继续复制该类别的文件到目标文件夹1
        if copied_counts_1[category] < copies_per_category:
            # 构建目标文件的完整路径
            destination_file_path_1 = os.path.join(destination_folder_1, filename)
            destination_file_path_2 = os.path.join(destination_folder_2, filename)
            # 复制文件到目标文件夹1
            shutil.copy(source_file_path, destination_file_path_1)
            shutil.copy(source_file_path, destination_file_path_2)
            print(f'已复制文件: {filename} 到类别: {category} 在目标文件夹1')
            # 更新复制计数
            copied_counts_1[category] += 1

        # # 判断是否继续复制该类别的文件到目标文件夹2
        # elif copied_counts_2[category] < copies_per_category:
        #     # 构建目标文件的完整路径
        #     destination_file_path_2 = os.path.join(destination_folder_2, filename)
        #     # 复制文件到目标文件夹2
        #     shutil.copy(source_file_path, destination_file_path_2)
        #     print(f'已复制文件: {filename} 到类别: {category} 在目标文件夹2')
        #     # 更新复制计数
        #     copied_counts_2[category] += 1

        else:
            print(f'类别: {category} 在两个目标文件夹中已达到复制次数上限，跳过: {filename}')

    else:
        print(f'文件名格式不正确，跳过: {filename}')

print('文件复制完成。')

# # 定义文件夹路径
# source_folder = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage1/Scribble'
# destination_folder = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage2/Scribble'
#
# # 确保目标文件夹存在
# os.makedirs(destination_folder, exist_ok=True)
#
# # 存储每个类别已有图片的数量和需要补齐的数量
# category_counts_source = defaultdict(int)
# category_counts_destination = defaultdict(int)
#
# # 统计第一个文件夹中每个类别已有的图片数量
# for filename in os.listdir(source_folder):
#     parts = filename.split('-')
#     if len(parts) >= 6:
#         category = parts[4] + '-' + parts[5]
#         category_counts_source[category] += 1
#         category_counts_destination[category] = 0
#
# # 统计第二个文件夹中每个类别需要补齐的图片数量
# for filename in os.listdir(destination_folder):
#     parts = filename.split('-')
#     if len(parts) >= 6:
#         category = parts[4] + '-' + parts[5]
#         category_counts_destination[category] += 1
#
# # 遍历第一个文件夹中的文件，随机选择符合条件的图片复制到第二个文件夹中
# for category, count_needed in category_counts_destination.items():
#     if count_needed < 3:
#         # 计算需要从第一个文件夹中复制的数量
#         copies_needed = 3 - count_needed
#         # 获取第一个文件夹中该类别已有的图片列表
#         category_images = []
#         for filename in os.listdir(source_folder):
#             parts = filename.split('-')
#             origin_category = parts[4] + '-' + parts[5]
#             if origin_category == category:
#                 category_images.append(filename)
#         # 随机选择复制的图片
#         images_to_copy = random.sample(category_images, min(copies_needed, len(category_images)))
#
#         # 复制图片到第二个文件夹中的相应类别
#         for filename in images_to_copy:
#             source_file_path = os.path.join(source_folder, filename)
#             destination_file_path = os.path.join(destination_folder, filename)
#             shutil.copy(source_file_path, destination_file_path)
#             print(f'已复制文件: {filename} 到类别: {category} 在目标文件夹')

# 文本生成阶段

# stage1 生成
import os
def clear_txt_file(file_path):
    # 以写模式打开文件，清空内容
    with open(file_path, 'w') as file:
        pass
# 指定你的图片文件夹路径
image_folder = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage1/Scribble'
# 获取文件夹中所有图片文件的文件名（不包括扩展名）
image_files = [os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
# 将文件名写入一个文本文件
output_file = "/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage1.txt"
clear_txt_file(output_file)
with open(output_file, 'w') as file:
    for image_file in image_files:
        file.write(image_file + '\n')
print(f"Image file names (without extensions) saved to {output_file}")

# stage2 生成
import itertools

# 读取txt文件并整理图片名字典
def process_image_names(filename):
    category_dict = {}

    with open(filename, 'r') as file:
        for line in file:
            image_name = line.strip()
            # 获取类别信息，假设类别信息是通过短横线分隔的，例如 image1-cat1.jpg
            parts = image_name.split('-')
            if len(parts) >= 5:
                category = parts[4].split('.')[0]  # 去除扩展名获取类别名
                if category in category_dict:
                    category_dict[category].append(image_name)
                else:
                    category_dict[category] = [image_name]

    return category_dict

def write_list_to_txt(filename, my_list):
    with open(filename, 'a') as file:
        for item in my_list:
            file.write(str(item) + ' ')
        file.write('\n')

# 输出整理后的结果
def print_category_combinations(category_dict, output_filename):
    with open(filename, 'r') as file:
        for line in file:
            image_name = line.strip()
            # 获取类别信息，假设类别信息是通过短横线分隔的，例如 image1-cat1.jpg
            parts = image_name.split('-')[4]
            image_cate = category_dict[parts]
            temp_image_cate = image_cate
            for img_name in image_cate:
                if img_name == image_name:
                    temp_image_cate.remove(image_name)
                    temp_image_cate.insert(0, image_name)
            write_list_to_txt(output_filename, temp_image_cate)
            print(temp_image_cate)

def remove_last_newline(filename):
    # 读取文件内容
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 删除最后一行末尾的换行符
    if lines:
        lines[-1] = lines[-1].rstrip('\n')

    # 将修改后的内容写回文件
    with open(filename, 'w') as file:
        file.writelines(lines)




# 示例用法
filename = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage1.txt'  # 替换为你的txt文件路径
output_filename = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage2.txt'
category_dict = process_image_names(filename)
clear_txt_file(output_filename)
print_category_combinations(category_dict, output_filename)
# 删除最后换行符
remove_last_newline(output_filename)