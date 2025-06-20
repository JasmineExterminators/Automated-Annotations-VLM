# prompt gemini directly to recognize text
# look for more robot datasets







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
from google.genai import types
from prompt_template import get_annotate_prompt, get_task_prompt

# get video path
if len(sys.argv) < 2:
     print(f"Usage: python {os.path.basename(__file__)} <VIDEOS_PATH>")
     sys.exit(1)

# Configuration
API_KEY = "AIzaSyDjnJusDy6ZyKhylNP-qot_ZgRSJOaoepo" # robyn's
FRAME_GAP = 20
VIDEOS_PATH = sys.argv[1]
MODEL = "gemini-2.5-pro-preview-03-25"
# MODEL = "gemini-2.5-flash-preview-05-20"
FPS = 20 #change to get it from the video lmao

# position of the label (previous, current) on each annotated frame
# will move later
PHOTO_X = 175
PHOTO_Y = 10

client = genai.Client(api_key= API_KEY) 

class Annotation(BaseModel):
    observation: str
    action: str
    reasoning: str
    summary: str

class Task(BaseModel):
    step_number: int
    name: str
    duration: int
    # might expand this

def video_to_pdf(
    video_path,
    pdf_path,
    frame_gap=20,
    photo_x=10,
    photo_y=10,
    photo_w=190
):
    """
    Converts a video to a PDF of frames.
    Args:
        video_path (str or Path): Path to the input video file.
        pdf_path (str or Path): Path to save the output PDF file.
        frame_gap (int): Number of frames to skip between each PDF page.
        photo_x (int): X position of the image in the PDF.
        photo_y (int): Y position of the image in the PDF.
        photo_w (int): Width of the image in the PDF.
    Returns:
        int: Number of frames processed (added to PDF).
    """
    import cv2
    from fpdf import FPDF
    import os
    import numpy as np
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    pdf = FPDF()
    demo_name = os.path.splitext(os.path.basename(str(video_path)))[0]
    processed_frames = 0
    print(f"Converting video {video_path} to pdf {pdf_path}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished video to pdf.")
            break
        if frame_count % frame_gap == 0:
            frame_filename = f"{demo_name}frame_{frame_count:04d}.jpg"
            cv2.imwrite(frame_filename, frame)
            pdf.add_page()
            time_seconds = frame_count / 20  # You may want to pass FPS as a parameter
            pdf.set_xy(photo_x, photo_y - 5)
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 10, f"Time: {time_seconds:.2f}s", ln=1)
            pdf.image(frame_filename, x=photo_x, y=photo_y + 15, w=photo_w)
            os.remove(frame_filename)
            processed_frames += 1
        frame_count += 1
    cap.release()
    pdf.output(str(pdf_path))
    return processed_frames

# initial prompt to gemini (run once per demo) to get a task list
def get_subtask_list(frame_filename, task_name, task_output):
    frame_upload = client.files.upload(file = frame_filename)
    prompt = get_task_prompt(task_name)
    response = client.models.generate_content(
        model=MODEL,
        contents=[frame_upload, prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": list[Task]
        }
    )
    if response.text:
        task_list = json.loads(response.text)
        # Save task list to a text file
        task_file_path = task_output + '/tasks.txt'
        with open(task_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Task List for {task_name}\n")
            f.write("=" * 50 + "\n\n")
            for task in task_list:
                f.write(f"{task["step_number"]}. {task['name']}\n")
                f.write(f"Task duration: {task['duration']}\n")
                f.write("-" * 30 + "\n")
    return task_list
    
# saves the final outputted json of observations, actions, reasonings, and summaries
def save_formatted_annotations(context, output_path):
    """
    Saves the annotation context to a JSON file with proper formatting.
    
    Args:
        context: List of annotation data
        output_path: Path where the JSON file will be saved
    """
    try:
        formatted_data = []
        for item in context:
            formatted_data.append(Annotation(
                observation=item[0]['observation'],
                action=item[0]['action'],
                reasoning=item[0]['reasoning'],
                summary=item[0]["summary"]
            ).model_dump())
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved JSON to {output_path}")
        return formatted_data
    except Exception as e:
        print(f"Error saving final annotations: {e}")
        return None

# prompts gemini for an annotation. run every frame_gap frames for each demo
def get_frame_annotation(prev_frame_uploaded, current_frame_uploaded, PROMPT, thoughts_file_path, frame_count):
    """
    Makes a Gemini API call to get annotations for a frame pair and processes the response.
    
    Args:
        prev_frame_uploaded: Uploaded previous frame
        current_frame_uploaded: Uploaded current frame
        PROMPT: The prompt to send to Gemini
        thoughts_file_path: Path to the thoughts file
        frame_count: Current frame number
    
    Returns:
        list: New context data if successful, None if failed
    """
    try:
        print(f"Making API call for frame {frame_count}...")
        response = client.models.generate_content(
            model=MODEL, 
            contents=[prev_frame_uploaded, current_frame_uploaded, PROMPT],
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[Annotation],
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True
                )
            )
        )
        
        print(f"Received response for frame {frame_count}")
        
        # Parse response and update context
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thought_text = f"Frame {frame_count}:\n{part.text}\n"
                # Append thought to the thoughts file
                with open(thoughts_file_path, 'a', encoding='utf-8') as thoughts_file:
                    thoughts_file.write(thought_text + "\n" + "-" * 50 + "\n\n")
            else:
                if response.text:
                    try:
                        new_context = json.loads(response.text)
                        if isinstance(new_context, list):
                            return new_context
                        else:
                            print(f"Warning: Unexpected response format: {response.text}")
                    except json.JSONDecodeError as e:
                        print(f"Error parsing Gemini response: {e}")
                else:
                    print("Warning: Empty response from Gemini")
                    
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    
    return None

# the main function that processes each demo (generates and saves annotations, renders video)
def process_demo(demo_path, task_name, VIDEOS_PATH):
    """
    Process a single demo video file, generating annotations and creating an annotated video.
    
    Args:
        demo_path: Path to the demo video file
        task_name: Name of the task being performed
        VIDEOS_PATH: Base path for output files
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    demo_name = os.path.splitext(os.path.basename(demo_path))[0]
    
    thoughts_file_path = Path(VIDEOS_PATH) / f"{demo_name}_thoughts.txt"
    with open(thoughts_file_path, 'w', encoding='utf-8') as thoughts_file:
        thoughts_file.write(f"Thought summaries for {demo_name}\n")
        thoughts_file.write("=" * 50 + "\n\n")
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
    first_frame_filename = os.path.join(VIDEOS_PATH, f"{demo_name}_frame_first.jpg")
    cv2.imwrite(first_frame_filename, first_frame)
    
    context = []  # Initialize as empty list instead of [{}]
    time_gap = (1 / FPS) * FRAME_GAP # time between frames in seconds
    task_list = get_subtask_list(first_frame_filename, task_name, VIDEOS_PATH)
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
            
            PROMPT = get_annotate_prompt(task_name, json.dumps([item[0]['summary'] for item in context]), time_gap, task_list, frame_count == 0, (frame_count / FPS))
            # print(PROMPT)                          
        
            prev_frame_filename = os.path.join(VIDEOS_PATH, f"{demo_name}_frame_prev_{frame_count}.jpg")
            current_frame_filename = os.path.join(VIDEOS_PATH, f"{demo_name}_frame_current_{frame_count}.jpg")
            
            # temporarily save frames
            cv2.imwrite(prev_frame_filename, prev_frame)
            cv2.imwrite(current_frame_filename, current_frame)
            
            # upload frames  
            prev_frame_uploaded = client.files.upload(file = prev_frame_filename)
            current_frame_uploaded = client.files.upload(file = current_frame_filename)
            
            new_annotation = get_frame_annotation(prev_frame_uploaded, current_frame_uploaded, PROMPT, thoughts_file_path, frame_count)
            if new_annotation is not None:
                context.append(new_annotation)
            
            # Clean up temporary files
            try:
                os.remove(current_frame_filename)
                os.remove(prev_frame_filename)
                
            except Exception as e:
                print(f"Warning: Error cleaning up temporary files: {e}")
            prev_frame = prev_temp
            cv2.putText(prev_frame, "Previous Frame", (PHOTO_X, PHOTO_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 4)
            cv2.putText(prev_frame, "Previous Frame", (PHOTO_X, PHOTO_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        frame_count += 1
        
    cap.release()
    os.remove(first_frame_filename)
    # Save final annotations
    output_annotations_path = Path(VIDEOS_PATH) / f"{demo_name}.json"
    formatted_data = save_formatted_annotations(context, output_annotations_path)
    if formatted_data is None:
        return False

    # Render and save the annotated video
    output_video_path = Path(VIDEOS_PATH) / f"{demo_name}_annotated.mp4"
    render_annotated_video(
        demo_path,
        output_video_path,
        formatted_data,
        time_gap
    )
    return True

# renders the annotated video with action and reasoning above the video
def render_annotated_video(
    video_path,
    output_path,
    annotations,
    time_gap,  # Time in seconds between annotation changes
    font=cv2.FONT_HERSHEY_PLAIN,
    font_scale=0.7,  # Reduced font size
    thickness=1,  # Reduced thickness for smaller text
):
    """
    Renders an annotated video with text overlays.
    Changes annotations every time_gap seconds.
    Layout: Video takes up bottom 2/3, annotation overlay takes up top 1/3.
    """
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

    # Open the input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'. Check path or file integrity.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    print(f"Input video properties: {width}x{height} @ {fps} FPS, {total_frames} frames ({duration_sec:.2f} seconds).")

    # Set output dimensions
    OUTPUT_FRAME_WIDTH = width * 3
    OUTPUT_FRAME_HEIGHT = height * 3

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (OUTPUT_FRAME_WIDTH, OUTPUT_FRAME_HEIGHT))
    if not out.isOpened():
        print(f"Error: Could not open video writer for '{output_path}'. Check codec or permissions.")
        cap.release()
        return

    current_annotation_index = 0
    last_change_time = 0
    frame_count = 0

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

    # Clean up
    cap.release()
    out.release()
    print(f"Video processing complete for '{output_path}'.")

def main():
    with os.scandir(VIDEOS_PATH) as demos:
        task_name = os.path.basename(VIDEOS_PATH)
        for demo in demos:
            if demo.name.endswith(".mp4") and demo.name.startswith("demo_") and demo.name[5:-4].isdigit():
                demo_path = Path(VIDEOS_PATH) / demo.name
                if not process_demo(demo_path, task_name, VIDEOS_PATH):
                    print(f"Failed to process demo: {demo.name}")

if __name__ == "__main__":
    main() 