import os
import shutil

# 定义文件夹路径
reference_folder = 'E:\Mr.Wu\dataset\Semi-CodDataset\pretrain\Pick1Image'
source_folder = 'E:\Mr.Wu\dataset\Semi-CodDataset\pretrain\GT'
destination_folder = 'E:\Mr.Wu\dataset\Semi-CodDataset\pretrain\Pick1GT'

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 获取参考文件夹中的所有文件名（不带后缀）
reference_files = {os.path.splitext(f)[0] for f in os.listdir(reference_folder)}

# 遍历源文件夹中的文件
for filename in os.listdir(source_folder):
    # 获取文件名（不带后缀）
    name_without_extension = os.path.splitext(filename)[0]

    # 如果文件名在参考文件夹中存在（不带后缀）
    if name_without_extension in reference_files:
        # 构建源文件和目标文件的完整路径
        source_file_path = os.path.join(source_folder, filename)
        destination_file_path = os.path.join(destination_folder, filename)

        # 复制文件
        shutil.copy(source_file_path, destination_file_path)
        print(f'已复制文件: {filename}')

print('文件复制完成。')
