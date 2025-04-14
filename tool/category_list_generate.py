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

# 示例用法
my_list = ['item1', 'item2', 'item3']
filename = 'output.txt'
write_list_to_txt(filename, my_list)

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
print_category_combinations(category_dict, output_filename)
# 删除最后换行符
remove_last_newline(output_filename)
