# batch mode / async python for gemini

from google import genai
from pydantic import BaseModel
import numpy as np
import cv2
import json
import os
from pathlib import Path
from fpdf import FPDF


 
client = genai.Client(api_key="AIzaSyA3JNDI0RArQ2V7X_eF_P6Y3DX8gP5hGDQ")
FRAME_GAP = 10
VIDEOS_PATH = "C:/Users/wuad3/Documents/CMU/Freshman Year/Research/SAMPLE"
PHOTO_X = 10
PHOTO_Y = 10
PHOTO_W = 190


# File tree 
# folder
#    name of task (folder)
#      videos


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
                    
                    print("converting video to pdf...")
                    while True:
                        # Capture frame-by-frame
                        ret, frame = cap.read()
                    
                        # if frame is read correctly ret is True
                        if not ret:
                            print("finished video to pdf.")
                            break
                        
                        if frame_count % FRAME_GAP == 0:
                            frame_filename = f"{demo_name}frame_{frame_count:04d}.jpg"
                            cv2.imwrite(frame_filename, frame)
                            pdf.add_page()
                            # Add time in seconds to the top-left corner before the image
                            time_seconds = frame_count / 20
                            pdf.set_xy(PHOTO_X, PHOTO_Y - 5)  # Slightly above the image
                            pdf.set_font("Arial", size=10)
                            pdf.cell(0, 10, f"Time: {time_seconds:.2f}s", ln=1)
                            pdf.image(frame_filename, x=PHOTO_X, y=PHOTO_Y + 15, w=PHOTO_W)
                            os.remove(frame_filename)
                            # print(f"Frame {frame_count} saved as {frame_filename}")
                        frame_count += 1
                    
                    # When everything done, release the capture
                    cap.release()

                    # Getting PDF
                    pdf_filename = f"{demo_name}_frames.pdf"
                    pdf.output(pdf_filename)
                    
        
                    # 2. Feed pdf, prompt to Gemini
                    input_pdf = client.files.upload(file = pdf_filename)
                    
                    # get the length of the video to help gemini
                    length = frame_count / 20
                    
                    prompt = f"""
                    First, examine the first page of the pdf provided. Describe the scene in front of you internally without outputting a response. When observing, pay careful attention to the the task name, {task.name}. Then, 
                    
                    segment the video provided in the pdf into detailed actions and detailed reasonings the robot is performing. Remember, the goal of the robot's task is: {task.name}.  You should record the observation of the scene (as the first field), the action the robot is performing, a reasoning for the action, a start time of the action, end time of the action, and duration of the action. The reasonings must be written in first person, thinking as if you are the robot.
                    
                    The first pdf named {demo_name}_frames.pdf shows all the frames of the state of the robot throughout a task being done. The left side shows the front view and the right side shows the view on the grippers of the robot. Each frame is spaced 0.05 seconds apart. This video has a length of {length} seconds.
                    
                    The action annotation is relatively fine-grained. For example, grasping is divided into 2 actions: reach, close the gripper. If the task description highlights spatial relationships, or if there are multiple objects from the same category, then your annotation should also contain these spatial / directional info, such as left / right, front / back. For the reasoning part, you only need to provide key steps. Focus on key visual features that help you identify the current situation. For example, the robot "is holding sth." or "has not reached sth." Remember that Gemini's visual understanding is worse than reasoning ability, so help Gemini more with visual info.
                                        
                 """
                    print(prompt)
                    # 3. save
  
                    class Annotation(BaseModel):
                        observation: str  # New field for an observation of the scene (now first)
                        action: str
                        reasoning: str
                        start: float
                        end: float
                        duration: float

                    print("sending to gemini...")
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[input_pdf, prompt],
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": list[Annotation]
                        },
                    )
                    print("gemini response received.")
                    
                    output_path = Path(VIDEOS_PATH) / task.name / f"{demo_name}.json"
                    
                    try:
                        # Assuming response.text contains the JSON string directly
                        data_from_gemini = json.loads(response.text)
                        
                        # Apply rounding and calculate duration for each item in the list
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
                            ).model_dump()) # .model_dump() to convert Pydantic model to dict
                            
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
