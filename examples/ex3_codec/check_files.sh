#!/bin/bash

dir_a="$1"
dir_b="$2"
file_extension="$3"

# 递归检查目录中的特定后缀文件，并进行重采样
check_files() {
    for file_a in "$1"/*.$file_extension; do
        file_b="${dir_b}/${file_a##*/}"
        if [[ ! -e "$file_b" ]]; then
            echo "文件 $file_a 在目录 B 中不存在。"
            # 运行 resample，将 A 目录中的文件通过 ffmpeg 重采样到 8k 并复制到 B 目录下
            ffmpeg -i "$file_a" -ar 8000 "${file_b}"
        fi
    done
}

# 检查输入参数个数是否正确
if [[ $# -ne 3 ]]; then
    echo "脚本需要三个参数：目录A，目录B和文件后缀。"
    exit 1
fi

# 检查目录 A 中的文件在目录 B 中是否存在，并进行重采样
check_files "$dir_a"

# 检查目录 B 中的文件在目录 A 中是否存在，并进行重采样
check_files "$dir_b"
