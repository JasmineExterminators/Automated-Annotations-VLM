# batch mode / async python for gemini
# fix json 
# better prompt
# match reference annotation with actually annnoated pdf


from google import genai
from pydantic import BaseModel
import numpy as np
import cv2
import json
import os
from pathlib import Path
from fpdf import FPDF
import tempfile
import glob
import io


 
client = genai.Client(api_key="AIzaSyA3JNDI0RArQ2V7X_eF_P6Y3DX8gP5hGDQ")
FRAME_GAP = 10
VIDEOS_PATH = "C:/Users/wuad3/Downloads/TEST"
PHOTO_X = 10
PHOTO_Y = 10
PHOTO_W = 190


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return root / name


with os.scandir(VIDEOS_PATH) as tasks:
    for task in tasks:
        task_dir = Path(VIDEOS_PATH) / task.name
        with os.scandir(task_dir) as demos:
            txt_file_name = next(f for f in os.listdir(Path(VIDEOS_PATH) / task.name) if f.endswith('.txt'))
            annotation_path = Path(VIDEOS_PATH) / task.name / txt_file_name
            for demo in demos:
                if demo.name.endswith(".mp4"):
                    demo_path = Path(VIDEOS_PATH) / task.name / demo.name
                    demo_name = os.path.splitext(os.path.basename(demo_path))[0]
                    # annotation_path = find(f"{task.name}_demo*", ANNOTATIONS_PATH)
                    # search_pattern = os.path.join(ANNOTATIONS_PATH, f"{task.name}_demo*.txt")
                    # annotation_path = glob.glob(search_pattern)[0]
                    # annotation_name = os.path.splitext(os.path.basename(annotation_path))[0]
                    # annotation_path = "C:/Users/wuad3/Downloads/TEST_ANNOTATIONS/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo_robyn_demo_0.txt"
                    # annotation_name = "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo_robyn_demo_0"

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
                            pdf.image(frame_filename, x=PHOTO_X, y=PHOTO_Y, w=PHOTO_W)
                            os.remove(frame_filename)
                            # print(f"Frame {frame_count} saved as {frame_filename}")
                        frame_count += 1
                    
                    # When everything done, release the capture
                    cap.release()

                    # Getting PDF
                    pdf_filename = f"{demo_name}_frames.pdf"
                    pdf.output(pdf_filename)
                    
        
                    # 2. Feed pdf, reference annotation, prompt to Gemini
                    input_pdf = client.files.upload(file = pdf_filename)
                    
                    # get the length of the video to help gemini
                    length = frame_count / 20
                    
                    # prompt = """These are frames of the state of the robot throughout a part of this task being done.
                    #             The left side shows the front view and the right side shows the view on the claw of the robot
                    #             The goal is to pick up the book and place it in the right compartment of the caddy
                    #             These frames are 0.4 seconds apart from each other.
                    #             Can you segment this task with action commands and its reasoning (written in first person), 
                    #             with reference to the frames, in order to determine the start and end times of an action? 
                    #             (don't need to explain every single frame, just segment it into actions as you see fit)"""
                    
                    prompt = f"""Can you segment the video provided in the pdf into detailed actions and detailed reasonings the robot is performing? You should record the action the robot is performing, a reasoning for the action, a start time of the action, end time of the action, and duration of the action.
                    
                    The pdf shows frames of the state of the robot throughout a part of a task being done. The left side shows the front view and the right side shows the view on the grippers of the robot. Each frame is spaced 0.05 seconds apart. This video has a length of {length} seconds.
                                       
                    The text file is the reference annotation of the task written by a human. Use this reference annotation to guide your style and formatting. The actions and reasonings should be more detailed and lengthy than the human reference. The reasonings must be written in first person, thinking as if you are the robot. 
                    
                    Note that the goal of this task is: {task.name}."""

                            
                    reference_annotation = client.files.upload(file = annotation_path)

                    # 3. save
  
                    class Annotation(BaseModel):
                        action: str
                        reasoning: str
                        start: float
                        end: float
                        duration: float

                    print("sending to gemini...")
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[input_pdf, prompt, reference_annotation],
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
