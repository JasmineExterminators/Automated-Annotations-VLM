import os

# Path to the folder containing all subfolders to be traversed
LIBERO_90_PATH = "C:/Users/wuad3/Documents/CMU/Freshman Year/Research/LIBERO-90-COLORED"  # <-- Set this to your actual path

# Names to keep (without extension)
KEEP_NAMES = {"demo_0", "demo_1", "demo_2"}

for root, dirs, files in os.walk(LIBERO_90_PATH):
    for file in files:
        name, ext = os.path.splitext(file)
        if name not in KEEP_NAMES:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}") 