import os

# 1. 请将这里的路径替换为您想操作的根文件夹
root_path = "/home/smiling/My_Codes/FakeFaceGen"

print(f"将在文件夹 '{root_path}' 及其所有子文件夹中，删除文件名包含 'zone' 的文件。")
print("警告：此操作不可撤销！请确保您已通过安全检查脚本确认过文件列表！")
input("按 Enter 键继续，或按 Ctrl+C 取消...") # 在执行前增加一个确认步骤

# 检查路径是否存在
if not os.path.isdir(root_path):
    print(f"错误：文件夹 '{root_path}' 不存在。")
else:
    try:
        delete_count = 0
        # 遍历所有文件夹和文件
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                # 检查文件名是否包含 "zone" (不区分大小写)
                if "zone" in filename.lower():
                    full_path = os.path.join(dirpath, filename)
                    try:
                        os.remove(full_path)
                        print(f"已删除: {full_path}")
                        delete_count += 1
                    except OSError as e:
                        print(f"错误：删除文件 {full_path} 时出错。原因: {e}")

        print(f"\n操作完成。总共删除了 {delete_count} 个文件。")

    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")