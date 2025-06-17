import os
import json
import cv2
import numpy as np
from pathlib import Path
from fpdf import FPDF
from google import genai
from pydantic import BaseModel

# Configuration
client = genai.Client(api_key="AIzaSyA3JNDI0RArQ2V7X_eF_P6Y3DX8gP5hGDQ")
FRAME_GAP = 10
VIDEOS_PATH = "C:/Users/wuad3/Documents/CMU/Freshman Year/Research/SAMPLE"
PHOTO_X = 10
PHOTO_Y = 10
PHOTO_W = 190

class Annotation(BaseModel):
    observation: str
    action: str
    reasoning: str
    start: float
    end: float
    duration: float

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
    font_scale=1.0,
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

def main():
    with os.scandir(VIDEOS_PATH) as tasks:
        for task in tasks:
            task_dir = Path(VIDEOS_PATH) / task.name
            with os.scandir(task_dir) as demos:
                for demo in demos:
                    if demo.name.endswith(".mp4"):
                        demo_path = Path(VIDEOS_PATH) / task.name / demo.name
                        demo_name = os.path.splitext(os.path.basename(demo_path))[0]
                        
                        # 1. Convert video frames to pdf
                        cap = cv2.VideoCapture(demo_path)
                        frame_count = 0
                        pdf = FPDF()
                        
                        print("Converting video to pdf...")
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                print("Finished video to pdf.")
                                break
                            
                            if frame_count % FRAME_GAP == 0:
                                frame_filename = f"{demo_name}frame_{frame_count:04d}.jpg"
                                cv2.imwrite(frame_filename, frame)
                                pdf.add_page()
                                time_seconds = frame_count / 20
                                pdf.set_xy(PHOTO_X, PHOTO_Y - 5)
                                pdf.set_font("Arial", size=10)
                                pdf.cell(0, 10, f"Time: {time_seconds:.2f}s", ln=1)
                                pdf.image(frame_filename, x=PHOTO_X, y=PHOTO_Y + 15, w=PHOTO_W)
                                os.remove(frame_filename)
                            frame_count += 1
                        
                        cap.release()
                        pdf_filename = f"{demo_name}_frames.pdf"
                        pdf.output(pdf_filename)
                        
                        # 2. Feed pdf, prompt to Gemini
                        input_pdf = client.files.upload(file=pdf_filename)
                        length = frame_count / 20
                        
                        prompt = f"""
                        First, examine the first page of the pdf provided. Describe the scene in front of you internally without outputting a response. When observing, pay careful attention to the the task name, {task.name}. Then, 
                        
                        segment the video provided in the pdf into detailed actions and detailed reasonings the robot is performing. Remember, the goal of the robot's task is: {task.name}.  You should record the observation of the scene (as the first field), the action the robot is performing, a reasoning for the action, a start time of the action, end time of the action, and duration of the action. The reasonings must be written in first person, thinking as if you are the robot.
                        
                        The first pdf named {demo_name}_frames.pdf shows all the frames of the state of the robot throughout a task being done. The left side shows the front view and the right side shows the view on the grippers of the robot. Each frame is spaced 0.05 seconds apart. This video has a length of {length} seconds.
                        
                        The action annotation is relatively fine-grained. For example, grasping is divided into 2 actions: reach, close the gripper. If the task description highlights spatial relationships, or if there are multiple objects from the same category, then your annotation should also contain these spatial / directional info, such as left / right, front / back. For the reasoning part, you only need to provide key steps. Focus on key visual features that help you identify the current situation. For example, the robot "is holding sth." or "has not reached sth." Remember that Gemini's visual understanding is worse than reasoning ability, so help Gemini more with visual info.
                        
                        Include more detailed information about the object's spatial relationships and the robot position. For instance, "I am positioned next to the OBJECT" and "the OBJECT is on the left side of the scene." There should be a paragraph of reasoning.
                        """
                        print(prompt)
                        
                        print("Sending to Gemini...")
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[input_pdf, prompt],
                            config={
                                "response_mime_type": "application/json",
                                "response_schema": list[Annotation]
                            },
                        )
                        print("Gemini response received.")
                        
                        output_path = Path(VIDEOS_PATH) / task.name / f"{demo_name}.json"
                        
                        try:
                            data_from_gemini = json.loads(response.text)
                            formatted_data = []
                            for item in data_from_gemini:
                                start_time = round(float(item['start']), 3)
                                end_time = round(float(item['end']), 3)
                                duration_time = round(end_time - start_time, 3)
                                formatted_data.append(Annotation(
                                    observation=item['observation'],
                                    action=item['action'],
                                    reasoning=item['reasoning'],
                                    start=start_time,
                                    end=end_time,
                                    duration=duration_time
                                ).model_dump())
                            
                            with open(output_path, "w", encoding="utf-8") as f:
                                json.dump(formatted_data, f, indent=4, ensure_ascii=False)
                            print(f"Successfully saved JSON to {output_path}")

                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON from Gemini response: {e}")
                            print(f"Gemini raw response text: {response.text}")
                        except Exception as e:
                            print(f"An unexpected error occurred: {e}")

                        # Clean up the generated PDF file
                        os.remove(pdf_filename)

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
                            OUTPUT_FRAME_HEIGHT
                        )

                        cap.release()
                        out.release()

                        print(f"Video processing complete for '{output_video_path}'.")

if __name__ == "__main__":
    main() 