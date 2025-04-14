import os


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
                print(f"Deleted file: {file_path}")
            else:
                print(f"Skipping non-file item: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


# 示例使用
folder_path = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage3/Scribble'  # 替换为你的文件夹路径
delete_all_files_in_folder(folder_path)
