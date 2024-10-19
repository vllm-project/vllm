import os
import sys
import shutil
import subprocess
import warnings

def main():
    # 获取 vllm 包的安装路径
    output = subprocess.run([sys.executable, "-m", "pip", "show", "vllm"],
                            capture_output=True)

    if output.returncode != 0:
        print("vllm 未安装")
        sys.exit(1)

    text = output.stdout.decode("utf-8")

    package_path = None
    for line in text.split("\n"):
        if line.startswith("Location: "):
            package_path = line.split(": ")[1]
            break

    if package_path is None:
        print("无法找到 vllm 包的安装路径")
        sys.exit(1)

    cwd = os.getcwd()

    if cwd == package_path:
        print("当前目录就是 vllm 包的安装目录，无需撤销开发模式")
        sys.exit(1)

    vllm_symlink_path = os.path.join(package_path, "vllm")
    vllm_backup_path = os.path.join(package_path, "vllm_pre_built")
    current_vllm_path = os.path.join(cwd, "vllm")

    # 检查是否在开发模式下
    if not os.path.islink(vllm_symlink_path):
        print(f"未处于开发模式：{vllm_symlink_path} 不是符号链接")
        sys.exit(1)

    if os.readlink(vllm_symlink_path) != current_vllm_path:
        print("当前目录不是源代码目录，无法撤销开发模式")
        sys.exit(1)

    # 解除符号链接
    print(f"解除符号链接 {vllm_symlink_path}")
    os.unlink(vllm_symlink_path)

    # 恢复备份的 vllm 目录
    if os.path.exists(vllm_backup_path):
        print(f"恢复备份的 vllm 目录，从 {vllm_backup_path} 到 {vllm_symlink_path}")
        os.rename(vllm_backup_path, vllm_symlink_path)
    else:
        print(f"备份目录 {vllm_backup_path} 不存在，无法恢复")
        sys.exit(1)

    # 可选：删除复制到当前目录的文件
    files_to_delete = [
        "vllm/_C.abi3.so",
        "vllm/_moe_C.abi3.so",
        "vllm/vllm_flash_attn/vllm_flash_attn_c.abi3.so",
        "vllm/vllm_flash_attn/flash_attn_interface.py",
        "vllm/vllm_flash_attn/__init__.py",
        "vllm/_version.py",
    ]

    for file in files_to_delete:
        file_path = os.path.join(cwd, file)
        if os.path.exists(file_path):
            print(f"删除文件 {file_path}")
            os.remove(file_path)

    print("已成功撤销开发模式设置")

if __name__ == "__main__":
    main()
