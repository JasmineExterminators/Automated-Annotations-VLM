import cv2
import numpy as np
import os
import json

def wrap_text(text, font, font_scale, thickness, max_width):
    """Splits text into lines so each line fits within max_width pixels."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        (w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if w > max_width and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)
    return lines

def overlay_text(
    frame_count,
    cap,
    out,
    fps, 
    width,
    height,
    total_frames,
    annotations,
    OUTPUT_FRAME_WIDTH,
    OUTPUT_FRAME_HEIGHT,
    font=cv2.FONT_HERSHEY_PLAIN,
    font_scale=1.0,  # Slightly larger for higher-res, but still small
    font_color=(0, 0, 0),
    thickness=2,
):
    """
    Overlays specified text onto an MP4 video using only OpenCV.
    Now overlays directly on the video, with a translucent white background for the annotation.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = frame_count / fps

        # Resize frame to output size (3x for higher resolution)
        if (frame.shape[1], frame.shape[0]) != (OUTPUT_FRAME_WIDTH, OUTPUT_FRAME_HEIGHT):
            frame = cv2.resize(frame, (OUTPUT_FRAME_WIDTH, OUTPUT_FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC)

        annot = annotations[0]
        action_words = annot["action"]
        reasoning_words = annot["reasoning"]
        annot_start_time = annot["start"]
        annot_end_time = annot["end"]
        annot_duration_time = annot["duration"]

        text_to_overlay = f"ACTION: {action_words}  REASONING: {reasoning_words}"

        # Calculate max text width (full width minus padding)
        max_text_width = OUTPUT_FRAME_WIDTH - 60  # 30px padding on each side
        lines = wrap_text(text_to_overlay, font, font_scale, thickness, max_text_width)
        (text_width, text_height), baseline = cv2.getTextSize(
            text_to_overlay, font, font_scale, thickness
        )
        line_height = text_height + baseline + 12
        total_text_height = line_height * len(lines)
        x, y = 30, 60  # 30px from left, 60px from top

        # Only use the top half of the screen for annotation
        max_annotation_height = OUTPUT_FRAME_HEIGHT // 2 - 40  # 40px padding from half
        if total_text_height > max_annotation_height:
            # If text is too tall, reduce font_scale (minimum 0.5)
            while total_text_height > max_annotation_height and font_scale > 0.5:
                font_scale -= 0.1
                lines = wrap_text(text_to_overlay, font, font_scale, thickness, max_text_width)
                (text_width, text_height), baseline = cv2.getTextSize(
                    text_to_overlay, font, font_scale, thickness
                )
                line_height = text_height + baseline + 12
                total_text_height = line_height * len(lines)

        if current_time_sec <= annot_end_time:
            # Draw translucent white rectangle as background for annotation (top half only)
            overlay = frame.copy()
            rect_top_left = (20, y - text_height - 20)
            rect_bottom_right = (OUTPUT_FRAME_WIDTH - 20, y + total_text_height + 20)
            if rect_bottom_right[1] > OUTPUT_FRAME_HEIGHT // 2:
                rect_bottom_right = (rect_bottom_right[0], OUTPUT_FRAME_HEIGHT // 2)
            cv2.rectangle(
                overlay,
                rect_top_left,
                rect_bottom_right,
                (255, 255, 255),
                thickness=-1
            )
            alpha = 0.7  # Transparency factor
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Draw each line of text
            for i, line in enumerate(lines):
                y_line = y + i * line_height
                if y_line > OUTPUT_FRAME_HEIGHT // 2 - 10:
                    break  # Don't draw past the top half
                cv2.putText(
                    frame,
                    line,
                    (x, y_line),
                    font,
                    font_scale,
                    font_color,
                    thickness,
                    cv2.LINE_AA
                )
            out.write(frame)
            frame_count += 1
        else:
            if len(annotations) != 1:
                annotations.pop(0)

if __name__ == "__main__":
    # Path to the folder containing all subfolders to be traversed
    LIBERO_90_PATH = "C:/Users/wuad3/Documents/CMU/Freshman Year/Research/SAMPLE"

    # Iterate through each folder (task folder)
    for root, dirs, files in os.walk(LIBERO_90_PATH):
        # Group files by their base name (without extension)
        file_groups = {}
        for file in files:
            name, ext = os.path.splitext(file)
            if name not in file_groups:
                file_groups[name] = []
            file_groups[name].append(file)

        # For each group, check if there is a matching video and JSON annotation
        for name, group_files in file_groups.items():
            video_file = None
            json_file = None
            for file in group_files:
                if file.endswith('.mp4'):
                    video_file = file
                elif file.endswith('.json'):
                    json_file = file

            if video_file and json_file:
                # Construct full paths
                video_path = os.path.join(root, video_file)
                json_path = os.path.join(root, json_file)
                output_path = os.path.join(root, f"{name}_annotated.mp4")

                # Open video file
                print(f"Starting video annotation overlay for '{video_path}'...")
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error: Could not open video file '{video_path}'. Check path or file integrity.")
                    continue

                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_sec = total_frames / fps
                print(f"Input video properties: {width}x{height} @ {fps} FPS, {total_frames} frames ({duration_sec:.2f} seconds).")

                OUTPUT_FRAME_WIDTH = width * 3  # 3x input video width
                OUTPUT_FRAME_HEIGHT = height * 3  # 3x input video height

                # Set up the video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
                out = cv2.VideoWriter(output_path, fourcc, fps, (OUTPUT_FRAME_WIDTH, OUTPUT_FRAME_HEIGHT))
                if not out.isOpened():
                    print(f"Error: Could not open video writer for '{output_path}'. Check codec or permissions.")
                    cap.release()
                    continue

                frame_count = 0

                with open(json_path, "r", encoding="utf-8") as f:
                    annotations = json.load(f)

                overlay_text(
                    frame_count,
                    cap,
                    out,
                    fps, 
                    width,
                    height,
                    total_frames,
                    annotations,
                    OUTPUT_FRAME_WIDTH,
                    OUTPUT_FRAME_HEIGHT
                )

                # Release everything when done
                cap.release()
                out.release()

                print(f"Video processing complete for '{output_path}'.")