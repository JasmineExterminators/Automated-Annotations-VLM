import cv2
import numpy as np

# Global variable for the video path
VIDEO_PATH = r"C:\Users\wuad3\Downloads\test_video.mp4"
OUTPUT_PATH = r"C:\Users\wuad3\Downloads\output_with_dot.mp4"

def create_dots_overlay(width, height, dots, dot_radius=15, font_scale=0.8, font_thickness=2):
    """
    Create a transparent overlay with all dots and numbers drawn.
    Returns a BGRA image (with alpha channel).
    """
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    for x, y, number, dot_color in dots:
        # Draw the semi-transparent dot (on alpha channel)
        cv2.circle(overlay, (x, y), dot_radius, dot_color[:3] + (0,), -1)  # Draw color only
        # Draw alpha channel for the dot
        alpha_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(alpha_mask, (x, y), dot_radius, dot_color[3], -1)
        overlay[:, :, 3] = np.maximum(overlay[:, :, 3], alpha_mask)
        # Draw the number (white, fully opaque)
        text = str(number)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = x - text_width // 2
        text_y = y + text_height // 2
        cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255, 255), font_thickness, cv2.LINE_AA)
    return overlay

def overlay_image_on_frame(frame, overlay_img):
    """
    Overlays an RGBA image (overlay_img) onto a BGR frame using the overlay's alpha channel.
    The overlay_img must be the same size as the frame.
    """
    if overlay_img.shape[2] == 3:
        # No alpha channel, treat as fully opaque
        overlay_alpha = np.ones(overlay_img.shape[:2], dtype=np.float32)
    else:
        overlay_alpha = overlay_img[:, :, 3] / 255.0
    overlay_rgb = overlay_img[:, :, :3]
    frame = frame.astype(np.float32)
    overlay_rgb = overlay_rgb.astype(np.float32)
    overlay_alpha = overlay_alpha[..., None]
    frame = (1 - overlay_alpha) * frame + overlay_alpha * overlay_rgb
    return frame.astype(np.uint8)


def save_transparent_overlay_with_dots(overlay_img, dots, output_path, dot_radius=15, font_scale=0.8, font_thickness=2, size=(224, 224)):
    """
    Save a transparent PNG (224x224) with the overlay image and the dots, preserving all original alpha/transparency qualities of the overlay image.
    """
    # Resize overlay image to target size
    overlay_img = cv2.resize(overlay_img, size, interpolation=cv2.INTER_AREA)
    # If overlay_img has no alpha, add fully opaque alpha
    if overlay_img.shape[2] == 3:
        overlay_img = np.concatenate([overlay_img, 255 * np.ones((size[1], size[0], 1), dtype=np.uint8)], axis=2)
    # Start with the overlay image as the base
    base = overlay_img.copy()
    # Create the dots overlay
    dots_overlay = create_dots_overlay(size[0], size[1], dots, dot_radius=dot_radius, font_scale=font_scale, font_thickness=font_thickness)
    # Alpha blend the dots overlay onto the base
    dots_alpha = dots_overlay[:, :, 3:4] / 255.0
    base_rgb = base[:, :, :3].astype(np.float32)
    dots_rgb = dots_overlay[:, :, :3].astype(np.float32)
    base[:, :, :3] = ((1 - dots_alpha) * base_rgb + dots_alpha * dots_rgb).astype(np.uint8)
    # Update alpha channel: max of base and dots overlay
    base[:, :, 3] = np.clip(base[:, :, 3] + dots_overlay[:, :, 3], 0, 255)
    # Save as PNG
    cv2.imwrite(output_path, base)

def save_dots_overlay(dots, output_path, dot_radius=15, font_scale=0.8, font_thickness=2, size=(224, 224)):
    """
    Save just the dots overlay as a transparent PNG (default 224x224).
    """
    dots_overlay = create_dots_overlay(size[0], size[1], dots, dot_radius=dot_radius, font_scale=font_scale, font_thickness=font_thickness)
    cv2.imwrite(output_path, dots_overlay)

if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # Define the dots to overlay: (x, y, number, dot_color)
    dots = [
        (20, 100, 1, (0, 0, 255, 128)),      # Red dot
        (150, 100, 1, (0, 0, 255, 128)),      # red dot
        (50, 25, 2, (255, 0, 0, 128)),      # Blue dot
        (150,25, 2, (255, 0, 0, 128)),      # Blue dot
    ]

    # Create the overlay once
    dots_overlay = create_dots_overlay(width, height, dots)
    overlay_rgb = dots_overlay[:, :, :3]
    overlay_alpha = dots_overlay[:, :, 3:4] / 255.0

    # Load and resize the image overlay
    image_overlay_path = r"C:\Users\wuad3\Downloads\heatmap_combined_all_dirs_combined_overlay.png"
    image_overlay = cv2.imread(image_overlay_path, cv2.IMREAD_UNCHANGED)
    image_overlay = cv2.resize(image_overlay, (width, height))
    # Make overlay more visible by increasing alpha
    if image_overlay.shape[2] == 4:
        # Boost alpha channel
        image_overlay = image_overlay.copy()
        image_overlay[:, :, 3] = np.clip(image_overlay[:, :, 3] * 2, 0, 255).astype(np.uint8)
    else:
        # Add alpha channel if missing
        alpha = np.full((height, width, 1), 200, dtype=np.uint8)
        image_overlay = np.concatenate([image_overlay, alpha], axis=2)

    # Save the first frame with overlays and dots as a 224x224 image
    # cap_for_image = cv2.VideoCapture(VIDEO_PATH)
    # ret, first_frame = cap_for_image.read()
    # if ret:
    #     save_image_with_overlay_and_dots(
    #         first_frame,
    #         image_overlay,
    #         dots,
    #         'output_image_224x224.png'
    #     )
    # cap_for_image.release()

    # Save transparent overlay with dots
    save_transparent_overlay_with_dots(
        image_overlay,
        dots,
        r'C:\Users\wuad3\Downloads\output_transparent_overlay_224x224.png'
    )

    # Save just the dots overlay as a transparent PNG
    save_dots_overlay(
        dots,
        r'C:\Users\wuad3\Downloads\output_dots_overlay_224x224.png'
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Blend the dots overlay onto the frame
        frame = overlay_image_on_frame(frame, image_overlay)
        frame = frame.astype(np.float32)
        frame = (1 - overlay_alpha) * frame + overlay_alpha * overlay_rgb
        frame = frame.astype(np.uint8)
        out.write(frame)
    cap.release()
    out.release() 