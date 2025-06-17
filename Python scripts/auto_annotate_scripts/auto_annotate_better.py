# To use this file, search for TODO and modfiy the lines accordingly. The available files to enter 
# are first_file_uploaded, prev_file_uploaded, next_file_uploaded, and PROMPT. The past history 
# is uploaded via the prompt as {json.dumps(context, indent=2)}.

import os
import json
import cv2
from pathlib import Path
from google import genai # import as pip install google-genai
from pydantic import BaseModel
import numpy as np
import sys


# get video path
# Ensure VIDEOS_PATH is provided via command-line
# if len(sys.argv) != 2:
#      print(f"Usage: python {os.path.basename(__file__)} <VIDEOS_PATH>")
#      sys.exit(1)


# Configuration
client = genai.Client(api_key="AIzaSyDjnJusDy6ZyKhylNP-qot_ZgRSJOaoepo") # robyn's
FRAME_GAP = 20
# VIDEOS_PATH = sys.argv[1]
VIDEOS_PATH = "C:/Users/wuad3/Documents/CMU/Freshman Year/Research/SAMPLE"
MODEL = "gemini-2.5-pro-preview-03-25"
# MODEL = "gemini-2.5-flash-preview-05-20"
FPS = 20 #change to get it from the video lmao


PHOTO_X = 175
PHOTO_Y = 10
class Annotation(BaseModel):
    observation: str
    action: str
    reasoning: str

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

def main():
    with os.scandir(VIDEOS_PATH) as tasks:
        for task in tasks:
            task_dir = Path(VIDEOS_PATH) / task.name
            if os.path.isdir(task_dir):
                with os.scandir(task_dir) as demos:
                    for demo in demos:
                        if demo.name.endswith(".mp4") and demo.name.startswith("demo_") and demo.name[5:-4].isdigit():
                            demo_path = Path(VIDEOS_PATH) / task.name / demo.name
                            demo_name = os.path.splitext(os.path.basename(demo_path))[0]
                            
                            cap = cv2.VideoCapture(demo_path)
                            frame_count = 0
                            ret, frame = cap.read()
                            
                            # initialize stuff
                            prev_frame = frame.copy() if ret else None
                            cv2.putText(prev_frame, "Previous Frame", (PHOTO_X, PHOTO_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 4)
                            cv2.putText(prev_frame, "Previous Frame", (PHOTO_X, PHOTO_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            
                            first_frame = frame.copy() if ret else None
                            cv2.putText(first_frame, "First Frame", (PHOTO_X, PHOTO_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 4)
                            cv2.putText(first_frame, "First Frame", (PHOTO_X, PHOTO_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            
                            
                            context = []  # Initialize as empty list instead of [{}]
                            time_gap = (1 / FPS) * FRAME_GAP # time between frames in seconds
                            print("reading video...")
                            while True:
                                ret, current_frame = cap.read()
                                
                                if not ret:
                                    print("Finished reading video.")
                                    break
                                if frame_count % FRAME_GAP == 0:
                                    print(f"Processing frame {frame_count}...")
                                    prev_temp = current_frame.copy() 
                                    cv2.putText(current_frame, "Current Frame", (PHOTO_X, PHOTO_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,0), 4)
                                    cv2.putText(current_frame, "Current Frame", (PHOTO_X, PHOTO_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                                    # TODO Change this numbered list.
                                    # 4. The previous context: {json.dumps(context, indent=2)}.
                                    # 1. The first frame of the video displaying the initial state of the scene.  
                                    PROMPT = f"""You are a robot performing the task {task.name}. You are given two files. 
                                    
                                    1. The first frame of the video displaying the initial state of the scene.                    
                                    2. The previous frame from {time_gap} seconds ago. 
                                    3. The current frame that depicts the current state of the scene. 
                                    4. The previous context: {json.dumps(context, indent=2)}.
                                
                                The name of the frame ("previous", "current", or "first") is provided in the top right corner of the image.
                                
                                WARNING: The previous context is not necessarily accurate. The context DOES NOT gurantee that a previous action has actually been completed. DO NOT depend solely on the previous context. Focus on the current frame and the observations you can make from it.                      
                                    
                                The left side shows the front view and the right side shows the view on the grippers of the robot. 

                            MISSION: Your mission is to generate a detailed action and reasoning for the robot to take in the current frame.
                            
                            1. Examine the previous and current frame. Infer what happened between the two frames and what is happening right now. When observing, pay careful attention to the task name, {task.name}. Note object spatial relationships and the robot position. 
                                                    
                            2. Think about your observations and the past context. Generate an action that the robot should take in the current frame as well as a detailed reasoning for the action. The action annotation is very fine-grained. For example, grasping is divided into 2 actions: reach, close the gripper. 
                            
                            If the task description highlights spatial relationships, or if there are multiple objects from the same category, then your action and reasoning should also contain these spatial / directional info, such as left / right, front / back. Focus on key visual features that help you identify the current situation. For example, the robot "is holding something." or "has not reached something." For instance, "lift the bowl upwards and to the left towards the stove."
                            
                            REMEMBER: the task name is {task.name}. 
                            
                            An example action is "reach for the black bowl by the white plate." An example reasoning is "I am reaching for the black bowl by the white plate because I need to pick it up and place it in the caddy. The bowl is on the left side of the plate, and I need to ensure I grasp it securely."
                            
                            IMPORTANT: It is imperative that you do not hallucinate actions or reasonings. For instance, closely examine the eye-in-hand view of the robot. If the robot is not grasping an object, do not annotate it as such. It is okay for intervals to be annotated similarly or to annotate intermediate actions like "continue holding" or "maintain position" if the robot is not performing a distinct action.

                            """
                            
                        
                            
                                    # print(PROMPT)
                                    prev_frame_filename = os.path.join(VIDEOS_PATH, f"{demo_name}_frame_prev.jpg")
                                    current_frame_filename = os.path.join(VIDEOS_PATH, f"{demo_name}_frame_current.jpg")
                                    first_frame_filename = os.path.join(VIDEOS_PATH, f"{demo_name}_frame_first.jpg")
                                    
                                    # overlay labels onto images
                                    
                                    
                                    
                                    
                                    
                                    
                                    cv2.imwrite(prev_frame_filename, prev_frame)
                                    cv2.imwrite(current_frame_filename, current_frame)
                                    cv2.imwrite(first_frame_filename, first_frame)
                                    
                                    # Save frames to temporary file   
                                    # TODO Delete what you don't use.                             
                                    prev_frame_uploaded = client.files.upload(file = prev_frame_filename)
                                    current_frame_uploaded = client.files.upload(file = current_frame_filename)
                                    first_frame_uploaded = client.files.upload(file = first_frame_filename)
                                    
                                    try:
                                        print(f"Making API call for frame {frame_count}...")
                                        # print(json.dumps(context))
                                        response = client.models.generate_content(
                                            model=MODEL, 
                                            contents=[first_frame_uploaded, prev_frame_uploaded, current_frame_uploaded, PROMPT], # TODO CHANGE THIS LINE
                                            config={
                                                "response_mime_type": "application/json",
                                                "response_schema": list[Annotation]
                                            }
                                        )
                                        
                                        print(f"Received response for frame {frame_count}")
                                        
                                        # Parse response and update context
                                        if response.text:
                                            try:
                                                new_context = json.loads(response.text)
                                                if isinstance(new_context, list):
                                                    context.append(new_context)
                                                else:
                                                    print(f"Warning: Unexpected response format: {response.text}")
                                            except json.JSONDecodeError as e:
                                                print(f"Error parsing Gemini response: {e}")
                                                # print(f"Raw response: {response.text}")
                                        else:
                                            print("Warning: Empty response from Gemini")
                                            
                                    except Exception as e:
                                        print(f"Error during Gemini API call: {e}")
                                        print(f"Error type: {type(e)}")
                                        import traceback
                                        print(f"Traceback: {traceback.format_exc()}")
                                        continue
                                    
                                    # Clean up temporary files
                                    try:
                                        os.remove(current_frame_filename)
                                        os.remove(prev_frame_filename)
                                        os.remove(first_frame_filename)
                                    except Exception as e:
                                        print(f"Warning: Error cleaning up temporary files: {e}")
                                    prev_frame = prev_temp
                                    cv2.putText(prev_frame, "Previous Frame", (PHOTO_X, PHOTO_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 4)
                                    cv2.putText(prev_frame, "Previous Frame", (PHOTO_X, PHOTO_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                                frame_count += 1
                                
                            
                            cap.release()
                            
                            # Save final annotations
                            output_path = Path(VIDEOS_PATH) / task.name / f"{demo_name}.json"
                            try:
                                formatted_data = []
                                for item in context:
                                    formatted_data.append(Annotation(
                                        observation=item[0]['observation'],
                                        action=item[0]['action'],
                                        reasoning=item[0]['reasoning']
                                    ).model_dump())
                                
                                with open(output_path, "w", encoding="utf-8") as f:
                                    json.dump(formatted_data, f, indent=4, ensure_ascii=False)
                                print(f"Successfully saved JSON to {output_path}")

                            except Exception as e:
                                print(f"Error saving final annotations: {e}")
                                # print(f"Context data: {context}")

                            # 3. Render the video with annotations
                            cap = cv2.VideoCapture(demo_path)
                            if not cap.isOpened():
                                print(f"Error: Could not open video file '{demo_path}'. Check path or file integrity.")
                                continue

                            fps = cap.get(cv2.CAP_PROP_FPS)
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            duration_sec = total_frames / fps
                            print(f"Input video properties: {width}x{height} @ {fps} FPS, {total_frames} frames ({duration_sec:.2f} seconds).")

                            OUTPUT_FRAME_WIDTH = width * 3
                            OUTPUT_FRAME_HEIGHT = height * 3

                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            output_video_path = Path(VIDEOS_PATH) / task.name / f"{demo_name}_annotated.mp4"
                            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (OUTPUT_FRAME_WIDTH, OUTPUT_FRAME_HEIGHT))
                            if not out.isOpened():
                                print(f"Error: Could not open video writer for '{output_video_path}'. Check codec or permissions.")
                                cap.release()
                                continue

                            frame_count = 0

                            with open(output_path, "r", encoding="utf-8") as f:
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
                                time_gap
                            )

                            cap.release()
                            out.release()

                            print(f"Video processing complete for '{output_video_path}'.")

if __name__ == "__main__":
    main() 