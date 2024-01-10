import os
import re

def get_last_number(path):
    # 通过斜杠分割路径，并取最后一个部分
    last_part = path.rsplit('/', 1)[-1]

    # 使用正则表达式提取数字
    match = re.search(r'(\d+)', last_part)

    return int(match.group()) if match else 0

def list_files(start_path):
    file_paths = []
    for root, dirs, files in os.walk(start_path):
        for file in sorted(files):
            if file.endswith('.png'):
                file_paths.append(os.path.join(root, file))
    return file_paths

def write_to_txt(file_paths, output_file):
    with open(output_file, 'w') as f:
        for path in file_paths:
            f.write(path + '\n')

if __name__=="__main__":
    # 训练域AB文件夹路径
    trainA_dir = '/home/jizefeng/share_datasets/ffhq/images'
    trainB_dir = '/home/jizefeng/share_datasets/ffhqr/ffhqr'
    # 指定输出文件路径
    outputA_txt_path = './ffhq_train.txt'
    outputB_txt_path = './ffhqr_train.txt'
    # 指定输出文件路径
    outputA_test_txt_path = './ffhq_test.txt'
    outputB_test_txt_path = './ffhqr_test.txt'

    # 获取文件夹及其子文件夹下所有文件路径
    filesA = list_files(trainA_dir)
    filesB = list_files(trainB_dir)
    filesB = sorted(filesB, key=get_last_number)
    # 将文件路径写入txt文件
    write_to_txt(filesA[:56000], outputA_txt_path)
    print(f"File paths written to: {outputA_txt_path}")
    write_to_txt(filesB[:56000], outputB_txt_path)
    print(f"File paths written to: {outputB_txt_path}")

    write_to_txt(filesA[56000:62000], outputA_test_txt_path)
    print(f"File paths written to: {outputA_test_txt_path}")
    write_to_txt(filesB[56000:62000], outputB_test_txt_path)
    print(f"File paths written to: {outputB_test_txt_path}")
