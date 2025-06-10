import os
import json
import cv2
import numpy as np
from pathlib import Path
from fpdf import FPDF
from google import genai
from pydantic import BaseModel
import base64

# Configuration
# client = genai.Client(api_key="AIzaSyAgWE8j36-f_NyfGPEHBMjgMXcMOMjaJjI") # cmu-aidm
client = genai.Client(api_key="AIzaSyA3JNDI0RArQ2V7X_eF_P6Y3DX8gP5hGDQ") # naveen
FRAME_GAP = 10
VIDEOS_PATH = "C:/Users/wuad3/Documents/CMU/Freshman Year/Research/SAMPLE"
MODEL = "gemini-2.0-flash"
FPS = 20


PHOTO_X = 10
PHOTO_Y = 10
PHOTO_W = 190
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
    time_gap,
    font=cv2.FONT_HERSHEY_PLAIN,
    font_scale=1.0,
    font_color=(0, 0, 0),
    thickness=2,
):
    """
    Overlays specified text onto an MP4 video using only OpenCV.
    Now overlays directly on the video, with a translucent white background for the annotation.
    """
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to output size
        if (frame.shape[1], frame.shape[0]) != (OUTPUT_FRAME_WIDTH, OUTPUT_FRAME_HEIGHT):
            frame = cv2.resize(frame, (OUTPUT_FRAME_WIDTH, OUTPUT_FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC)

        # Calculate which annotation to show based on current time
        current_time = frame_count / fps
        annotation_index = int(current_time / time_gap)
        
        if annotation_index < len(annotations):
            annot = annotations[annotation_index]
            action_words = annot["action"]
            reasoning_words = annot["reasoning"]
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

def main():
    with os.scandir(VIDEOS_PATH) as tasks:
        for task in tasks:
            task_dir = Path(VIDEOS_PATH) / task.name
            with os.scandir(task_dir) as demos:
                for demo in demos:
                    if demo.name.endswith(".mp4") and demo.name.startswith("demo_") and demo.name[5:-4].isdigit():
                        demo_path = Path(VIDEOS_PATH) / task.name / demo.name
                        demo_name = os.path.splitext(os.path.basename(demo_path))[0]
                        
                        cap = cv2.VideoCapture(demo_path)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration_sec = total_frames / FPS
                        frame_count = 0
                        ret, frame = cap.read()
                        
                        # initialize stuff
                        prev_frame = frame.copy() if ret else None
                        first_frame = frame.copy() if ret else None
                        context = []  # Initialize as empty list instead of [{}]
                        time_gap = (1 / FPS) * FRAME_GAP # time between frames in seconds
                        time_in_sec = 0
                        print("reading video...")
                        while True:
                            ret, next_frame = cap.read()
                            if not ret:
                                print("Finished reading video.")
                                break
                            
                            if frame_count % FRAME_GAP == 0:
                                print(f"Processing frame {frame_count}...")
                                
                                PROMPT = f"""You are a robot performing the task {task.name}. You are given three files. 
                                1. The previous frame from {time_gap} seconds ago. 
                                2. The current view of the scene. 
                                3. The previous context: {json.dumps(context, indent=2)}.
                                
                            The left side shows the front view and the right side shows the view on the grippers of the robot. 

                        MISSION: Your mission is to generate a detailed action and reasoning for the robot to take in the current frame.

                        DO NOT begin generating anything until instructed to do so.
                        
                        1. Examine the previous and current frame. Infer what happened between the two frames and what is happening right now. When observing, pay careful attention to the task name, {task.name}. Note object spatial relationships and the robot position. 
                                                
                        2. Think about your observations and the past context. Generate an action that the robot should take in the current frame as well as a detailed reasoning for the action. The action annotation is fine-grained. For example, grasping is divided into 2 actions: reach, close the gripper. If the task description highlights spatial relationships, or if there are multiple objects from the same category, then your action and reasoning should also contain these spatial / directional info, such as left / right, front / back. Focus on key visual features that help you identify the current situation. For example, the robot "is holding something." or "has not reached something." 
                        
                        REMEMBER: the task name is {task.name}. 
                        
                        An example action is "reach for the black bowl by the white plate." An example reasoning is "I am reaching for the black bowl by the white plate because I need to pick it up and place it in the caddy. The bowl is on the left side of the plate, and I need to ensure I grasp it securely."
                        
                        IMPORTANT: It is imperative that you do not hallucinate actions or reasonings. For instance, closely examine the eye-in-hand view of the robot. If the robot is not grasping an object, do not annotate it as such. 

                        """
                                # print(PROMPT)
                                prev_frame_filename = f"{demo_name}_frame_prev_{frame_count:04d}.jpg"
                                next_frame_filename = f"{demo_name}_frame_next_{frame_count:04d}.jpg"
                                first_frame_filename = f"{demo_name}_frame_first.jpg"
                                cv2.imwrite(prev_frame_filename, prev_frame)
                                cv2.imwrite(next_frame_filename, next_frame)
                                cv2.imwrite(first_frame_filename, first_frame)
                                
                                # Save frames to temporary file                                
                                prev_frame_uploaded = client.files.upload(file = prev_frame_filename)
                                next_frame_uploaded = client.files.upload(file = next_frame_filename)
                                first_frame_uploaded = client.files.upload(file = first_frame_filename)
                                
                                try:

                                    print(f"Making API call for frame {frame_count}...")
                                    print(json.dumps(context))
                                    response = client.models.generate_content(
                                        model=MODEL, 
                                        contents=[first_frame_uploaded, prev_frame_uploaded, next_frame_uploaded, PROMPT],
                                        config={
                                            "response_mime_type": "application/json",
                                            "response_schema": list[Annotation]
                                        }
                                    )
                                    
                                    print(f"Received response for frame {frame_count}")
                                    print(f"Response text: {response.text}")
                                    
                                    # Parse response and update context
                                    if response.text:
                                        try:
                                            new_context = json.loads(response.text)
                                            if isinstance(new_context, list):
                                                context.append(new_context)
                                                print(f"Now there are {len(context)} annotations")
                                            else:
                                                print(f"Warning: Unexpected response format: {response.text}")
                                        except json.JSONDecodeError as e:
                                            print(f"Error parsing Gemini response: {e}")
                                            print(f"Raw response: {response.text}")
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
                                    os.remove(next_frame_filename)
                                    os.remove(prev_frame_filename)
                                except Exception as e:
                                    print(f"Warning: Error cleaning up temporary files: {e}")
                                time_in_sec += time_gap
                            frame_count += 1
                            prev_frame = next_frame.copy() 
                        
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
                            print(f"Context data: {context}")

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