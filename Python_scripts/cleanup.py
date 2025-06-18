from pathlib import Path

# Path to the folder containing all subfolders to be traversed
FOLDER_PATH = Path(r"C:/Users/wuad3/Documents/CMU/Freshman_Year/Research/Automated-Annotations-VLM/Python_scripts/small_sample_of_videos")

# Exact filenames to keep
KEEP_FILES = {"demo_0.mp4"}

# Delete all files not in KEEP_FILES
for file_path in FOLDER_PATH.rglob("*"):
    if file_path.is_file() and file_path.name not in KEEP_FILES:
        try:
            file_path.unlink()
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

# Optional: delete empty directories
for dir_path in sorted(FOLDER_PATH.rglob("*"), key=lambda p: -p.as_posix().count("/")):
    if dir_path.is_dir() and not any(dir_path.iterdir()):
        try:
            dir_path.rmdir()
            print(f"Removed empty directory: {dir_path}")
        except Exception as e:
            print(f"Failed to remove directory {dir_path}: {e}")