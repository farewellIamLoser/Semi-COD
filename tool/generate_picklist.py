import os

# 指定你的图片文件夹路径
image_folder = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage1 - 副本/Scribble'

# 获取文件夹中所有图片文件的文件名（不包括扩展名）
image_files = [os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

# 将文件名写入一个文本文件
output_file = "/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage1.txt"
with open(output_file, 'w') as file:
    for image_file in image_files:
        file.write(image_file + '\n')

print(f"Image file names (without extensions) saved to {output_file}")

# # 指定你的图片文件夹路径
# image_folder = '/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage2/Scribble'
#
# # 获取文件夹中所有图片文件的文件名（不包括扩展名）
# image_files = [os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
#
# # 将文件名写入一个文本文件
# output_file = "/mnt/e/Mr.Wu/dataset/Semi-CodDataset/stage2.txt"
# with open(output_file, 'w') as file:
#     for image_file in image_files:
#         file.write(image_file + '\n')
#
# print(f"Image file names (without extensions) saved to {output_file}")


