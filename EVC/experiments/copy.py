import os, sys
import shutil

excludes = [
    "lightning_logs",
    "log",
    "experiments",
    "__pycache__",
    ".vscode",
    ".gitignore"
]

def copy_files_and_folders(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for item in os.listdir(src_dir):
        if item in excludes:
            continue

        src_item = os.path.join(src_dir, item)
        dst_item = os.path.join(dst_dir, item)

        if os.path.isfile(src_item) and src_item.endswith(".py"):
            shutil.copy2(src_item, dst_item)
        elif os.path.isdir(src_item):
            copy_files_and_folders(src_item, dst_item)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python copy.py <destination_dir>")
        sys.exit(1)

    dst_dir = sys.argv[1]
    copy_files_and_folders("../", dst_dir)

