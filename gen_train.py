import os

def generate_new_test_txt(sequences_dir, output_file):
    """
    生成新的 test.txt，每行包含一个子文件夹的路径。
    
    :param sequences_dir: sequences 文件夹的路径，例如 "data/vimeo_septuplet/sequences/"
    :param output_file: 要生成的 test.txt 文件路径，例如 "data/vimeo_septuplet/test_new.txt"
    """
    with open(output_file, 'w') as f:
        for sequence_id in sorted(os.listdir(sequences_dir)):
            sequence_path = os.path.join(sequences_dir, sequence_id)
            if not os.path.isdir(sequence_path):
                continue
            for subfolder_id in sorted(os.listdir(sequence_path)):
                subfolder_path = os.path.join(sequence_path, subfolder_id)
                if not os.path.isdir(subfolder_path):
                    continue
                # 检查子文件夹中是否包含 im1.png 到 im7.png
                images = [f"im{i}.png" for i in range(1, 8)]
                if all(os.path.isfile(os.path.join(subfolder_path, img)) for img in images):
                    # 写入相对于 sequences_dir 的路径
                    relative_path = os.path.join(sequence_id, subfolder_id)
                    f.write(relative_path + '\n')

if __name__ == '__main__':
    sequences_dir = "data/vimeo_septuplet/sequences/"
    output_file = "data/vimeo_septuplet/test_tree.txt"
    generate_new_test_txt(sequences_dir, output_file)
    print(f"新的 test.txt 已生成到 {output_file}")
