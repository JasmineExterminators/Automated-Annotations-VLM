# prompt gemini directly to recognize text
# look for more robot
# test on recovery videos

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
from fpdf import FPDF
from google.genai import types
from prompt_template import get_annotate_prompt, get_task_prompt

# get video path
if len(sys.argv) < 2:
     print(f"Usage: python {os.path.basename(__file__)} <VIDEOS_PATH>")
     sys.exit(1)

# Configuration
API_KEY = "AIzaSyDjnJusDy6ZyKhylNP-qot_ZgRSJOaoepo" # robyn's
FPS = 20 # change to get from video lmao
FRAME_GAP = 10
VIDEOS_PATH = sys.argv[1]
MODEL = "gemini-2.5-pro-preview-05-06"
# MODEL = "gemini-2.5-flash-preview-05-20"


client = genai.Client(api_key= API_KEY) 

class Annotation(BaseModel):
    start: float
    end: float
    action: str
    reasoning: str

class Task(BaseModel):
    step_number: int
    name: str
    duration: int
    # might expand this

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

def video_to_pdf(demo_path, output_dir, frame_gap, photo_w, fps=20):
    """
    Converts a video to a PDF of frames, saving the PDF in output_dir.
    Returns the path to the generated PDF and the number of frames processed.
    The image is centered in the middle of the page, with the time stamp above it.
    """
    demo_name = os.path.splitext(os.path.basename(demo_path))[0]
    cap = cv2.VideoCapture(demo_path)
    frame_count = 0
    pdf = FPDF()
    print("Converting video to pdf...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished video to pdf.")
            break
        if frame_count % frame_gap == 0:
            frame_filename = os.path.join(output_dir, f"{demo_name}frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            pdf.add_page()
            time_seconds = frame_count / fps
            # Center the image and time stamp
            page_width = pdf.w
            page_height = pdf.h
            image_x = (page_width - photo_w) / 2
            # Estimate image height based on aspect ratio of the frame
            img = cv2.imread(frame_filename)
            if img is not None:
                img_height, img_width = img.shape[:2]
                aspect_ratio = img_height / img_width
                photo_h = photo_w * aspect_ratio
            else:
                photo_h = photo_w  # fallback
            image_y = (page_height - photo_h) / 2
            time_text = f"Time: {time_seconds:.2f}s"
            pdf.set_font("Arial", size=20)
            time_text_width = pdf.get_string_width(time_text)
            time_x = (page_width - time_text_width) / 2
            time_y = image_y - 10  # 10 px above the image
            if time_y < 0:
                time_y = 0
            pdf.set_xy(time_x, time_y)
            pdf.cell(time_text_width, 10, time_text, ln=1)
            pdf.image(frame_filename, x=image_x, y=image_y, w=photo_w)
            os.remove(frame_filename)
        frame_count += 1
    cap.release()
    pdf_filename = os.path.join(output_dir, f"{demo_name}_frames.pdf")
    pdf.output(pdf_filename)
    return pdf_filename, frame_count

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

    # Convert video to PDF
    pdf_filename, total_frames = video_to_pdf(
        demo_path,
        VIDEOS_PATH,
        FRAME_GAP,
        224,  # Use 190 as default width for now
        FPS
    )
    
    # Upload PDF 
    input_pdf = client.files.upload(file=pdf_filename)
    length = total_frames / FPS
    
    # get prompt from prompt_template
    prompt = get_annotate_prompt(task_name, length)
    
    # create thoughts file
    thoughts_path = Path(VIDEOS_PATH) / f"{demo_name}_thoughts.txt"
    with open(thoughts_path, 'w', encoding='utf-8') as thoughts_file:
        thoughts_file.write(f"Prompt for Gemini: {prompt}")
        thoughts_file.write(f"=" * 50 + "\n\n")
        thoughts_file.write(f"Thought summaries for {demo_name}\n")
        thoughts_file.write("=" * 50 + "\n\n")
    print("Sending to Gemini...")
    
    # prompts gemini
    response = client.models.generate_content(
        model=MODEL,
        contents=[input_pdf, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[Annotation],
            thinking_config=types.ThinkingConfig(
                include_thoughts=True
            )
        )
    )
    print("Gemini response received.")
    # Write prompt and Gemini thoughts to thoughts file
    with open(thoughts_path, 'a', encoding='utf-8') as thoughts_file:
        # Extract and write thoughts from Gemini response
        if hasattr(response, 'candidates') and response.candidates:
            for part in response.candidates[0].content.parts:
                if getattr(part, 'thought', False):
                    thoughts_file.write("Gemini Thought:\n")
                    thoughts_file.write(str(part.text) + "\n" + ("-" * 50) + "\n\n")
    output_annotations_path = Path(VIDEOS_PATH) / f"{demo_name}.json"
    try:
        data_from_gemini = json.loads(response.text)
        formatted_data = []
        for item in data_from_gemini:
            start_time = round(float(item['start']), 3)
            end_time = round(float(item['end']), 3)
            duration_time = round(end_time - start_time, 3)
            formatted_data.append(Annotation(
                action=item['action'],
                reasoning=item['reasoning'],
                start=start_time,
                end=end_time,
                duration=duration_time
            ).model_dump())
        with open(output_annotations_path, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved JSON to {output_annotations_path}")
    except Exception as e:
        print(f"Error saving final annotations: {e}")
        return False
    finally:
        # Clean up the generated PDF file
        try:
            # os.remove(pdf_filename) 
            print("DID NOT DELETE PDF")
        except Exception as e:
            print(f"Warning: Error cleaning up PDF file: {e}")
    # Render and save the annotated video
    output_video_path = Path(VIDEOS_PATH) / f"{demo_name}_annotated.mp4"
    render_annotated_video(
        demo_path,
        output_video_path,
        formatted_data
    )
    return True

# renders the annotated video with action and reasoning above the video
def render_annotated_video(
    video_path,
    output_path,
    annotations,
    font=cv2.FONT_HERSHEY_PLAIN,
    font_scale=0.7,  # Reduced font size
    thickness=1,  # Reduced thickness for smaller text
):
    """
    Renders an annotated video with text overlays.
    Displays the annotation whose start <= current_time_sec < end for each frame.
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

        # Find the annotation for the current time
        current_annotation = None
        for annot in annotations:
            if annot["start"] <= current_time_sec < annot["end"]:
                current_annotation = annot
                break

        # Create a blank frame for the new layout
        combined_frame = np.zeros((OUTPUT_FRAME_HEIGHT, OUTPUT_FRAME_WIDTH, 3), dtype=np.uint8)
        # Resize the video frame to fit the bottom section
        video_frame = cv2.resize(frame, (video_width, video_height), interpolation=cv2.INTER_CUBIC)
        # Place the video frame in the bottom section
        combined_frame[overlay_height:, :] = video_frame

        # Create the overlay section (top 1/3)
        overlay = np.ones((overlay_height, OUTPUT_FRAME_WIDTH, 3), dtype=np.uint8) * 255  # White background

        if current_annotation is not None:
            action_words = current_annotation["action"]
            reasoning_words = current_annotation["reasoning"]
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