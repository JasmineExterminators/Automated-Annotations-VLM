import cv2
import numpy as np
import sys

VIDEOS_PATH = sys.argv[1]

def overlay_images(base_img_path, overlay_img_path, output_path, target_size=(224, 224)):
    """
    Overlay overlay_img on top of base_img using alpha blending and save the result.
    Both images are resized to target_size before blending.
    """
    base_img = cv2.imread(base_img_path, cv2.IMREAD_UNCHANGED)
    overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)

    # Resize both images to target_size
    base_img = cv2.resize(base_img, target_size)
    overlay_img = cv2.resize(overlay_img, target_size)

    # If base has no alpha, add fully opaque alpha
    if base_img.shape[2] == 3:
        base_img = np.concatenate([base_img, 255 * np.ones((base_img.shape[0], base_img.shape[1], 1), dtype=np.uint8)], axis=2)
    # If overlay has no alpha, add fully opaque alpha
    if overlay_img.shape[2] == 3:
        overlay_img = np.concatenate([overlay_img, 255 * np.ones((overlay_img.shape[0], overlay_img.shape[1], 1), dtype=np.uint8)], axis=2)

    # Alpha blending
    overlay_alpha = overlay_img[:, :, 3:4] / 255.0
    base_alpha = base_img[:, :, 3:4] / 255.0
    out_rgb = (1 - overlay_alpha) * base_img[:, :, :3] + overlay_alpha * overlay_img[:, :, :3]
    out_alpha = np.clip(base_img[:, :, 3:4] + overlay_img[:, :, 3:4], 0, 255)
    out_img = np.concatenate([out_rgb, out_alpha], axis=2).astype(np.uint8)

    cv2.imwrite(output_path, out_img)

if __name__ == "__main__":
    # Example usage
    base_img_path = r"C:\Users\wuad3\Downloads\heatmap_combined_all_dirs_combined_overlay.png"
    overlay_img_path = r"C:\Users\wuad3\Downloads\output_dots_overlay_224x224.png"
    output_path = r"C:\Users\wuad3\Downloads\output_overlayed.png"
    overlay_images(base_img_path, overlay_img_path, output_path, target_size=(224, 224)) 