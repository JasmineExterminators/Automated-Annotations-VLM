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
    time_gap,  # Time in seconds between annotation changes
    font=cv2.FONT_HERSHEY_PLAIN,
    font_scale=0.7,  # Reduced font size
    thickness=1,  # Reduced thickness for smaller text
):
    """
    Overlays specified text onto an MP4 video using only OpenCV.
    Changes annotations every time_gap seconds.
    Layout: Video takes up bottom 2/3, annotation overlay takes up top 1/3.
    """
    current_annotation_index = 0
    last_change_time = 0

    # Calculate dimensions for the new layout
    overlay_height = OUTPUT_FRAME_HEIGHT // 3
    video_height = OUTPUT_FRAME_HEIGHT - overlay_height
    video_width = OUTPUT_FRAME_WIDTH

    # Define colors for action and reasoning
    action_color = (0, 0, 255)  # Red in BGR
    reasoning_color = (255, 0, 0)  # Blue in BGR

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = frame_count / fps

        # Check if it's time to change the annotation
        if current_time_sec - last_change_time >= time_gap:
            current_annotation_index = (current_annotation_index + 1) % len(annotations)
            last_change_time = current_time_sec

        # Create a blank frame for the new layout
        combined_frame = np.zeros((OUTPUT_FRAME_HEIGHT, OUTPUT_FRAME_WIDTH, 3), dtype=np.uint8)
        
        # Resize the video frame to fit the bottom section
        video_frame = cv2.resize(frame, (video_width, video_height), interpolation=cv2.INTER_CUBIC)
        
        # Place the video frame in the bottom section
        combined_frame[overlay_height:, :] = video_frame

        # Create the overlay section (top 1/3)
        overlay = np.ones((overlay_height, OUTPUT_FRAME_WIDTH, 3), dtype=np.uint8) * 255  # White background

        annot = annotations[current_annotation_index]
        action_words = annot["action"]
        reasoning_words = annot["reasoning"]

        # Prepare action and reasoning text separately
        action_text = f"ACTION: {action_words}"
        reasoning_text = f"REASONING: {reasoning_words}"

        # Calculate max text width (full width minus padding)
        max_text_width = OUTPUT_FRAME_WIDTH - 60  # 30px padding on each side
        
        # Wrap each text separately
        action_lines = wrap_text(action_text, font, font_scale, thickness, max_text_width)
        reasoning_lines = wrap_text(reasoning_text, font, font_scale, thickness, max_text_width)
        
        # Calculate text dimensions
        (_, text_height), baseline = cv2.getTextSize(action_text, font, font_scale, thickness)
        line_height = text_height + baseline + 8  # Reduced spacing between lines
        
        # Calculate total height for both sections
        total_action_height = line_height * len(action_lines)
        total_reasoning_height = line_height * len(reasoning_lines)
        total_text_height = total_action_height + total_reasoning_height + 10  # 10px gap between sections
        
        # Center the text vertically in the overlay section
        start_y = (overlay_height - total_text_height) // 2 + text_height
        x = 30  # 30px from left

        # Draw action lines
        for i, line in enumerate(action_lines):
            y_line = start_y + i * line_height
            cv2.putText(
                overlay,
                line,
                (x, y_line),
                font,
                font_scale,
                action_color,
                thickness,
                cv2.LINE_AA
            )

        # Draw reasoning lines
        for i, line in enumerate(reasoning_lines):
            y_line = start_y + total_action_height + 10 + i * line_height  # 10px gap after action
            cv2.putText(
                overlay,
                line,
                (x, y_line),
                font,
                font_scale,
                reasoning_color,
                thickness,
                cv2.LINE_AA
            )

        # Place the overlay section in the top portion
        combined_frame[:overlay_height, :] = overlay

        out.write(combined_frame)
        frame_count += 1

if __name__ == "__main__":
    # Path to the folder containing all subfolders to be traversed
    LIBERO_90_PATH = "C:/Users/wuad3/Documents/CMU/Freshman Year/Research/test"

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
                    OUTPUT_FRAME_HEIGHT,
                    time_gap=1.0  # Change annotation every 1 second
                )

                # Release everything when done
                cap.release()
                out.release()

                print(f"Video processing complete for '{output_path}'.")