import os
import json
import cv2
from pathlib import Path
from google import genai as genai
from pydantic import BaseModel
import numpy as np
import time
from PIL import Image # Import the Pillow library

# --- Configuration ---
# TODO: Securely manage your API key instead of hardcoding it.
# Consider using environment variables or a secret management tool.
API_KEY = "YOUR_API_KEY" # <--- IMPORTANT: Replace with your actual key
client = genai.Client(api_key=API_KEY)

# TODO: Adjust these paths and parameters as needed for your environment.
VIDEOS_PATH = Path("C:/Users/wuad3/Documents/CMU/Freshman Year/Research/SAMPLE")
MODEL = "gemini-1.5-flash-preview-0514" # Using a fast and capable model
FRAME_GAP = 20
FPS = 20

class Annotation(BaseModel):
    observation: str
    action: str
    reasoning: str

def wrap_text(text, font, font_scale, thickness, max_width):
    """Splits text into lines so each line fits within max_width pixels."""
    words = text.split()
    if not words:
        return []
    
    lines = []
    current_line = words[0]
    for word in words[1:]:
        test_line = f"{current_line} {word}"
        (w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if w > max_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    lines.append(current_line)
    return lines

def main():
    """
    Main function to process all videos in the specified directory.
    This revised workflow operates in a single pass to optimize performance.
    """
    for task_dir in VIDEOS_PATH.iterdir():
        if not task_dir.is_dir():
            continue
            
        for demo_path in task_dir.glob("demo_*.mp4"):
            if not demo_path.stem.startswith("demo_") or not demo_path.stem[5:].isdigit():
                continue

            print(f"\n--- Processing video: {demo_path.name} in task: {task_dir.name} ---")

            cap = cv2.VideoCapture(str(demo_path))
            if not cap.isOpened():
                print(f"Error: Could not open video file '{demo_path}'.")
                continue

            # --- Video Properties ---
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # --- Output Video Setup ---
            output_video_path = task_dir / f"{demo_path.stem}_annotated.mp4"
            output_width, output_height = width * 2, height * 2 
            video_height = output_height * 2 // 3
            overlay_height = output_height - video_height

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (output_width, output_height))
            if not out.isOpened():
                print(f"Error: Could not open video writer for '{output_video_path}'.")
                cap.release()
                continue
            
            # --- Frame and Annotation Initialization ---
            frame_count = 0
            all_annotations = []
            current_annotation = {"action": "Initializing...", "reasoning": "Waiting for the first set of frames to process."}
            
            ret, first_frame = cap.read()
            if not ret:
                print("Error: Could not read the first frame.")
                cap.release()
                out.release()
                continue
            
            first_frame_pil = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
            
            prev_frame = first_frame.copy()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # --- Main Processing Loop (Single Pass) ---
            while True:
                ret, current_frame = cap.read()
                if not ret:
                    break

                if frame_count % FRAME_GAP == 0:
                    print(f"Analyzing frame {frame_count}...")
                    
                    prev_frame_pil = Image.fromarray(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB))
                    current_frame_pil = Image.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
                    
                    prompt = f"""You are a robot analyzing a video for the task: {task_dir.name}.
You are given three images: the first frame, a previous frame, and the current frame.
Based on these, provide a concise 'action' and 'reasoning' for the robot's next move.
"""
                    try:
                        # **CORRECTION**: Removed the 'generation_config' dictionary and passed
                        # its contents as direct keyword arguments to the method.
                        response = client.models.generate_content(
                            model=MODEL,
                            contents=[first_frame_pil, prev_frame_pil, current_frame_pil, prompt],
                            response_mime_type="application/json",
                            response_schema=Annotation
                        )
                        new_annotation = json.loads(response.text)
                        all_annotations.append(new_annotation)
                        current_annotation = new_annotation
                        print(f"  - Action for frame {frame_count}: {current_annotation['action']}")

                    except Exception as e:
                        print(f"Error during API call for frame {frame_count}: {e}")

                    prev_frame = current_frame.copy()

                # --- Video Rendering (for every frame) ---
                combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                video_frame_resized = cv2.resize(current_frame, (output_width, video_height), interpolation=cv2.INTER_AREA)
                combined_frame[overlay_height:, :] = video_frame_resized
                overlay = np.full((overlay_height, output_width, 3), 255, dtype=np.uint8)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale, thickness, padding = 0.9, 1, 20
                max_text_width = output_width - (2 * padding)
                line_spacing = 10

                action_lines = wrap_text(f"Action: {current_annotation['action']}", font, font_scale, thickness, max_text_width)
                reasoning_lines = wrap_text(f"Reasoning: {current_annotation['reasoning']}", font, font_scale, thickness, max_text_width)
                
                y_pos = padding + 25
                for line in action_lines:
                    text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                    cv2.putText(overlay, line, (padding, y_pos), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
                    y_pos += text_size[1] + 5

                y_pos += line_spacing
                for line in reasoning_lines:
                    text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                    cv2.putText(overlay, line, (padding, y_pos), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
                    y_pos += text_size[1] + 5
                
                combined_frame[:overlay_height, :] = overlay
                out.write(combined_frame)
                frame_count += 1

            # --- Cleanup and Finalization ---
            print(f"Finished processing. Releasing resources for {demo_path.name}.")
            cap.release()
            out.release()

            output_json_path = task_dir / f"{demo_path.stem}_annotations.json"
            try:
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(all_annotations, f, indent=4, ensure_ascii=False)
                print(f"Successfully saved annotations to {output_json_path}")
            except Exception as e:
                print(f"Error saving final annotations: {e}")

            print(f"--- Completed: {output_video_path.name} ---")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")