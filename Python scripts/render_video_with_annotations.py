import cv2
import numpy as np
import os
import json

def overlay_text(
    frame_count,
    cap,
    out,
    fps, 
    width,
    height,
    total_frames,
    duration_sec,
    text_to_overlay,
    start_time_sec,  # When text appears (seconds)
    end_time_sec,    # When text disappears (seconds)
    font=cv2.FONT_HERSHEY_PLAIN,
    font_scale=0.5,
    font_color=(0, 255, 255),  # BGR format: (Blue, Green, Red) - Yellow
    thickness=1,
):
    """
    Overlays specified text onto an MP4 video using only OpenCV.

    Args:
        input_video_path (str): Path to the input MP4 video file.
        output_video_path (str): Path to save the new MP4 video file with text.
        text_to_overlay (str): The text string to overlay.
        start_time_sec (float): Time in seconds when the text should appear.
        end_time_sec (float): Time in seconds when the text should disappear.
        font (int): OpenCV font type (e.g., cv2.FONT_HERSHEY_SIMPLEX).
        font_scale (float): Scale factor of the font.
        font_color (tuple): BGR color tuple (e.g., (255, 255, 255) for white).
        thickness (int): Thickness of the text lines.
    """
    while True:
        ret, frame = cap.read()

        if not ret: # hopefully not error and signifies end of vid
            break

        current_time_sec = frame_count / fps

        # Check if current time is within annot time
        if start_time_sec <= current_time_sec <= end_time_sec:
            # Calculate text size to help with positioning
            (text_width, text_height), baseline = cv2.getTextSize(
                text_to_overlay, font, font_scale, thickness
            )

            # Position text top left
            x, y = 10, text_height + 10 # 10 pixels from left, 10 pixels below text height

            # Draw main text
            cv2.putText(
                frame, # Frame is already BGR from cap.read()
                text_to_overlay,
                (x, y),
                font,
                font_scale,
                font_color,
                thickness,
                cv2.LINE_AA # Anti-aliasing for smoother text
            )

        out.write(frame) # Write the modified frame to the output video

        frame_count += 1

    

# def wrap_text(text, font, )



if __name__ == "__main__":
    G_ANNOTATIONS_PATH = "C:/Users/cajas/Downloads/demo_0.json"
    OG_VIDEO_PATH = "C:/Users/cajas/Downloads/demo_0(17).mp4"
    ANNOTED_VIDEO_PATH = "C:/Users/cajas/Downloads/demo_0(18).mp4"
    
    # Open vid file
    print(f"Starting video annotation overlay for '{OG_VIDEO_PATH}'...")
    cap = cv2.VideoCapture(OG_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{OG_VIDEO_PATH}'. Check path or file integrity.")
    
    # Get vid properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    print(f"Input video properties: {width}x{height} @ {fps} FPS, {total_frames} frames ({duration_sec:.2f} seconds).")

    # Set up the video writer (set up out vid)
    # Define the codec and create VideoWriter object
    # For .mp4, common codecs are 'mp4v' or 'XVID' (if available).
    # 'mp4v' is generally more compatible for MP4.
    # On some systems, 'avc1' (H.264) might work if ffmpeg is properly linked.
    # If this fails, try 'XVID'.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
    out = cv2.VideoWriter(ANNOTED_VIDEO_PATH, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for '{ANNOTED_VIDEO_PATH}'. Check codec or permissions.")
        cap.release()

    frame_count = 0

    with open(G_ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    for annot in annotations:
        action_words = annot["action"]
        reasoning_words = annot["reasoning"]
        annot_start_time = annot["start"]
        annot_end_time = annot["end"]
        annot_duration_time = annot["duration"]

        text_to_overlay = f"ACTION: {action_words}  REASONING: {reasoning_words}"

        overlay_text(
            frame_count,
            cap,
            out,
            fps, 
            width,
            height,
            total_frames,
            duration_sec,
            text_to_overlay,
            annot_start_time,
            annot_end_time)


    # Release everything when done
    cap.release()
    out.release()

    print(f"Video processing complete for '{ANNOTED_VIDEO_PATH}'.")