import os
import shutil


def filter_files(folder_A, folder_B):
    # 获取文件夹 A 和 B 中的文件列表
    files_A = os.listdir(folder_A)
    files_B = os.listdir(folder_B)

    # 提取文件名（不包含路径）
    filenames_A = set(os.path.splitext(file)[0] for file in files_A)
    filenames_B = set(os.path.splitext(file)[0] for file in files_B)

    # 找出需要保留的文件名
    files_to_keep = filenames_A.intersection(filenames_B)

    # 移动或删除不需要的文件
    for file in files_A:
        filename, extension = os.path.splitext(file)
        if filename not in files_to_keep:
            file_path = os.path.join(folder_A, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed {file_path}")
            else:
                print(f"Could not find {file_path} to remove")


if __name__ == "__main__":
    folder_A = 'E:\Mr.Wu\dataset\CodDataset\semi-test\Image'  # 替换为文件夹 A 的路径
    folder_B = 'E:\Mr.Wu\dataset\CodDataset\semi-test\Scribble'  # 替换为文件夹 B 的路径

    filter_files(folder_A, folder_B)
    print('finish')