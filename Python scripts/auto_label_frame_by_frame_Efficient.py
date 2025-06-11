# To use this file, search for TODO and modfiy the lines accordingly. The available files to enter 
# are first_file_uploaded, prev_file_uploaded, next_file_uploaded, and PROMPT. The past history 
# is uploaded via the prompt as {json.dumps(context, indent=2)}.

import os
import json
import cv2
import numpy as np
from pathlib import Path
from google import genai # import as pip install google-genai
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time
from collections import defaultdict
import asyncio
from typing import List, Dict, Tuple
import tempfile
import shutil

# Configuration
client = genai.Client(api_key="AIzaSyDjnJusDy6ZyKhylNP-qot_ZgRSJOaoepo") # robyn's
FRAME_GAP = 20
VIDEOS_PATH = "C:/Users/wuad3/Documents/CMU/Freshman Year/Research/SAMPLE"
MODEL = "gemini-2.5-pro-preview-03-25"
FPS = 20
BATCH_SIZE = 8  # Increased batch size for more parallel processing
MAX_WORKERS = 8  # Increased number of parallel workers
MAX_VIDEO_WORKERS = 8  # Increased number of parallel video workers
GEMINI_BATCH_SIZE = 16  # Number of frames to send to Gemini at once

# Thread-safe structures
context_lock = threading.Lock()
frame_queue = Queue()
video_queue = Queue()
context = defaultdict(list)

# Create a temporary directory for frame files
TEMP_DIR = tempfile.mkdtemp()
print(f"Created temporary directory: {TEMP_DIR}")

PHOTO_X = 10
PHOTO_Y = 10
PHOTO_W = 190
class Annotation(BaseModel):
    observation: str
    action: str
    reasoning: str

class FrameBatch:
    def __init__(self, task_name: str, demo_name: str, frame_count: int):
        self.task_name = task_name
        self.demo_name = demo_name
        self.frame_count = frame_count
        self.frames = []
        self.results = []
        self.temp_files = []  # Keep track of temporary files

    def add_frame(self, first_frame, prev_frame, next_frame):
        self.frames.append((first_frame, prev_frame, next_frame))

    def is_full(self):
        return len(self.frames) >= GEMINI_BATCH_SIZE

    def cleanup(self):
        """Clean up all temporary files associated with this batch"""
        for file_list in self.temp_files:
            for file_path in file_list:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Error cleaning up temporary file {file_path}: {e}")

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

async def process_gemini_batch(batch: FrameBatch, time_gap: float) -> List[Dict]:
    """Process a batch of frames with Gemini API"""
    try:
        # Save all frames in the batch
        frame_files = []
        for i, (first_frame, prev_frame, next_frame) in enumerate(batch.frames):
            unique_id = f"{batch.task_name}_{batch.demo_name}_{batch.frame_count + i * FRAME_GAP}"
            first_frame_filename = os.path.join(TEMP_DIR, f"temp_{unique_id}_first.jpg")
            prev_frame_filename = os.path.join(TEMP_DIR, f"temp_{unique_id}_prev.jpg")
            next_frame_filename = os.path.join(TEMP_DIR, f"temp_{unique_id}_current.jpg")
            
            # Save frames with error handling
            try:
                cv2.imwrite(first_frame_filename, first_frame)
                cv2.imwrite(prev_frame_filename, prev_frame)
                cv2.imwrite(next_frame_filename, next_frame)
                frame_files.append((first_frame_filename, prev_frame_filename, next_frame_filename))
            except Exception as e:
                print(f"Error saving frame files for {unique_id}: {e}")
                continue
        
        if not frame_files:
            print(f"No valid frames to process for batch {batch.demo_name}")
            return []
        
        # Upload all frames
        uploaded_files = []
        for first_file, prev_file, next_file in frame_files:
            try:
                first_uploaded = client.files.upload(file=first_file)
                prev_uploaded = client.files.upload(file=prev_file)
                next_uploaded = client.files.upload(file=next_file)
                uploaded_files.append((first_uploaded, prev_uploaded, next_uploaded))
            except Exception as e:
                print(f"Error uploading files for {batch.demo_name}: {e}")
                continue
        
        if not uploaded_files:
            print(f"No valid uploaded files for batch {batch.demo_name}")
            return []
        
        # Create prompts for all frames
        prompts = []
        for i in range(len(uploaded_files)):
            prompt = f"""You are a robot performing the task {batch.task_name}. You are given four files. 
            
            1. The first frame of the video displaying the initial state of the scene.                                                            
            2. The previous frame from {time_gap} seconds ago. 
            3. The current view of the scene. 
            4. The previous context: {json.dumps(context[batch.demo_name], indent=2)}.
            
            The left side shows the front view and the right side shows the view on the grippers of the robot. 

            MISSION: Your mission is to generate a detailed action and reasoning for the robot to take in the current frame.
            
            1. Examine the previous and current frame. Infer what happened between the two frames and what is happening right now. When observing, pay careful attention to the task name, {batch.task_name}. Note object spatial relationships and the robot position. 
                                                
            2. Think about your observations and the past context. Generate an action that the robot should take in the current frame as well as a detailed reasoning for the action. The action annotation is very fine-grained. For example, grasping is divided into 2 actions: reach, close the gripper. 
            
            If the task description highlights spatial relationships, or if there are multiple objects from the same category, then your action and reasoning should also contain these spatial / directional info, such as left / right, front / back. Focus on key visual features that help you identify the current situation. For example, the robot "is holding something." or "has not reached something." For instance, "lift the bowl upwards and to the left towards the stove."
            
            REMEMBER: the task name is {batch.task_name}. 
            
            An example action is "reach for the black bowl by the white plate." An example reasoning is "I am reaching for the black bowl by the white plate because I need to pick it up and place it in the caddy. The bowl is on the left side of the plate, and I need to ensure I grasp it securely."
            
            IMPORTANT: It is imperative that you do not hallucinate actions or reasonings. For instance, closely examine the eye-in-hand view of the robot. If the robot is not grasping an object, do not annotate it as such. It is okay for intervals to be annotated similarly or to annotate intermediate actions like "continue holding" or "maintain position" if the robot is not performing a distinct action.
            """
            prompts.append(prompt)
        
        # Process all frames in parallel
        tasks = []
        for i, (first_uploaded, prev_uploaded, next_uploaded) in enumerate(uploaded_files):
            task = client.models.generate_content(
                model=MODEL,
                contents=[first_uploaded, prev_uploaded, next_uploaded, prompts[i]],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": list[Annotation]
                }
            )
            tasks.append(task)
        
        # Wait for all responses
        responses = await asyncio.gather(*tasks)
        
        # Process responses
        results = []
        for response in responses:
            if response and hasattr(response, 'text') and response.text:
                try:
                    new_context = json.loads(response.text)
                    if isinstance(new_context, list) and len(new_context) > 0:
                        # Ensure we have the expected structure
                        if isinstance(new_context[0], dict) and all(key in new_context[0] for key in ['observation', 'action', 'reasoning']):
                            results.append(new_context)
                        else:
                            print(f"Warning: Unexpected response structure: {new_context}")
                except json.JSONDecodeError as e:
                    print(f"Error parsing Gemini response: {e}")
                except Exception as e:
                    print(f"Error processing response: {e}")
        
        return results
        
    except Exception as e:
        print(f"Error processing Gemini batch: {e}")
        return []
    finally:
        # Clean up temporary files
        for first_file, prev_file, next_file in frame_files:
            try:
                if os.path.exists(first_file):
                    os.remove(first_file)
                if os.path.exists(prev_file):
                    os.remove(prev_file)
                if os.path.exists(next_file):
                    os.remove(next_file)
            except Exception as e:
                print(f"Warning: Error cleaning up temporary files: {e}")

def process_video(task_name: str, demo_path: str, demo_name: str):
    """Process a single video file"""
    try:
        print(f"Starting processing of video: {demo_name}")
        cap = cv2.VideoCapture(demo_path)
        frame_count = 0
        ret, frame = cap.read()
        
        if not ret:
            print(f"Error: Could not read video file {demo_path}")
            return
            
        # Initialize frames
        first_frame = frame.copy()
        prev_frame = frame.copy()
        time_gap = (1 / FPS) * FRAME_GAP
        time_in_sec = 0
        
        # Initialize batch processing
        current_batch = FrameBatch(task_name, demo_name, frame_count)
        
        while True:
            ret, next_frame = cap.read()
            if not ret:
                # Process remaining frames
                if current_batch.frames:
                    results = asyncio.run(process_gemini_batch(current_batch, time_gap))
                    with context_lock:
                        if results:  # Only extend if we have results
                            context[demo_name].extend(results)
                    current_batch.cleanup()
                break
            
            if frame_count % FRAME_GAP == 0:
                current_batch.add_frame(first_frame.copy(), prev_frame.copy(), next_frame.copy())
                
                # Process batch when it's full
                if current_batch.is_full():
                    results = asyncio.run(process_gemini_batch(current_batch, time_gap))
                    with context_lock:
                        if results:  # Only extend if we have results
                            context[demo_name].extend(results)
                    current_batch.cleanup()
                    current_batch = FrameBatch(task_name, demo_name, frame_count + FRAME_GAP)
                
                time_in_sec += time_gap
            
            frame_count += 1
            prev_frame = next_frame.copy()
        
        cap.release()
        
        # Save final annotations
        output_path = Path(VIDEOS_PATH) / task_name / f"{demo_name}.json"
        try:
            formatted_data = []
            for item in context[demo_name]:
                if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
                    formatted_data.append(Annotation(
                        observation=item[0].get('observation', ''),
                        action=item[0].get('action', ''),
                        reasoning=item[0].get('reasoning', '')
                    ).model_dump())
            
            if formatted_data:  # Only save if we have data
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(formatted_data, f, indent=4, ensure_ascii=False)
                print(f"Successfully saved JSON to {output_path}")
            else:
                print(f"No valid annotations to save for {demo_name}")
            
        except Exception as e:
            print(f"Error saving final annotations for {demo_name}: {e}")
            return
        
        # Render the video with annotations
        if not formatted_data:  # Skip rendering if no annotations
            print(f"No annotations to render for {demo_name}")
            return
            
        cap = cv2.VideoCapture(demo_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{demo_path}' for rendering.")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        OUTPUT_FRAME_WIDTH = width * 3
        OUTPUT_FRAME_HEIGHT = height * 3
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = Path(VIDEOS_PATH) / task_name / f"{demo_name}_annotated.mp4"
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (OUTPUT_FRAME_WIDTH, OUTPUT_FRAME_HEIGHT))
        
        if not out.isOpened():
            print(f"Error: Could not open video writer for '{output_video_path}'.")
            cap.release()
            return
            
        frame_count = 0
        
        overlay_text(
            frame_count,
            cap,
            out,
            fps,
            width,
            height,
            total_frames,
            formatted_data,  # Use formatted_data instead of loading from file
            OUTPUT_FRAME_WIDTH,
            OUTPUT_FRAME_HEIGHT,
            time_gap
        )
        
        cap.release()
        out.release()
        print(f"Video processing complete for '{output_video_path}'")
        
    except Exception as e:
        print(f"Error processing video {demo_name}: {e}")
    finally:
        # Ensure video capture is released
        if 'cap' in locals() and cap is not None:
            cap.release()
        if 'out' in locals() and out is not None:
            out.release()

def main():
    try:
        # Create a list of all videos to process
        videos_to_process = []
        with os.scandir(VIDEOS_PATH) as tasks:
            for task in tasks:
                task_dir = Path(VIDEOS_PATH) / task.name
                with os.scandir(task_dir) as demos:
                    for demo in demos:
                        if demo.name.endswith(".mp4") and demo.name.startswith("demo_") and demo.name[5:-4].isdigit():
                            demo_path = Path(VIDEOS_PATH) / task.name / demo.name
                            demo_name = os.path.splitext(os.path.basename(demo_path))[0]
                            videos_to_process.append((task.name, str(demo_path), demo_name))
        
        # Process all videos in parallel
        with ThreadPoolExecutor(max_workers=MAX_VIDEO_WORKERS) as executor:
            futures = []
            for video in videos_to_process:
                future = executor.submit(process_video, *video)
                futures.append(future)
            
            # Wait for all videos to be processed
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing video: {e}")
        
        print("All videos processed successfully!")
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"Cleaned up temporary directory: {TEMP_DIR}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")

if __name__ == "__main__":
    main() 