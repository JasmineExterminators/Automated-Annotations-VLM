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
                    print(annotation_path)
                    # annotation_path = find(f"{task.name}_demo*", ANNOTATIONS_PATH)
                    # search_pattern = os.path.join(ANNOTATIONS_PATH, f"{task.name}_demo*.txt")
                    # annotation_path = glob.glob(search_pattern)[0]
                    # annotation_name = os.path.splitext(os.path.basename(annotation_path))[0]
                    # annotation_path = "C:/Users/wuad3/Downloads/TEST_ANNOTATIONS/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo_robyn_demo_0.txt"
                    # annotation_name = "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo_robyn_demo_0"

                    # # 1. Convert video frames to pdf
                    # cap = cv2.VideoCapture(demo_path)

                    # frame_count = 0
                    # pdf = FPDF()

                    # while True:
                    #     # Capture frame-by-frame
                    #     ret, frame = cap.read()
                    
                    #     # if frame is read correctly ret is True
                    #     if not ret:
                    #         print("Vid ended")
                    #         break
                        
                    #     if frame_count % FRAME_GAP == 0:
                    #         frame_filename = f"{demo_name}frame_{frame_count:04d}.jpg"
                    #         cv2.imwrite(frame_filename, frame)
                    #         pdf.add_page()
                    #         pdf.image(frame_filename, x=PHOTO_X, y=PHOTO_Y, w=PHOTO_W)
                    #         os.remove(frame_filename)
                    #         # print(f"Frame {frame_count} saved as {frame_filename}")
                    #     frame_count += 1
                    
                    # # When everything done, release the capture
                    # cap.release()

                    # # Getting PDF
                    # pdf_filename = f"{demo_name}_frames.pdf"
                    # pdf.output(pdf_filename)
                    

                    # # 2. Feed pdf, reference annotation, prompt to Gemini
                    # input_pdf = client.files.upload(file = pdf_filename)
                    # prompt = """These are frames of the state of the robot throughout a part of this task being done.
                    #             The left side shows the front view and the right side shows the view on the claw of the robot
                    #             The goal is to pick up the book and place it in the right compartment of the caddy
                    #             These frames are 0.4 seconds apart from each other.
                    #             Can you segment this task with action commands and its reasoning (written in first person), 
                    #             with reference to the frames, in order to determine the start and end times of an action? 
                    #             (don't need to explain every single frame, just segment it into actions as you see fit)
                    #         """
                    # reference_annotation = client.files.upload(file = annotation_path)

                    # # 3. save
                    # const schema = {
                    #     type: SchemaType.ARRAY,
                    #     items: {
                    #         type: SchemaType.OBJECT,
                    #         properties: {
                    #         action: { type: SchemaType.STRING, description: "action robot is performing", nullable: false },
                    #         reasoning: { type: SchemaType.STRING, description: "reasoning for the current action", nullable: false },
                    #         start: { type: SchemaType.DOUBLE, description: "start time of annotation", nullable: false },
                    #         end: { type: SchemaType.NUMBER, description: "IMDb rating of the movie", nullable: false },
                    #         },
                    #         required: ["title", "director", "year", "imdbRating"],
                    #     },
                    #     };


                    # response = client.models.generate_content(
                    #     model="gemini-2.0-flash",
                    #     contents=[input_pdf, prompt, reference_annotation],
                    #     config={
                    #         "response_mime_type": "application/json",
                    #         "response_schema": list[Annotation]
                    #     },
                    # )
                    
                    # print(response)
                    # # 4. Save the response
                    # output_path = Path(VIDEOS_PATH) / task.name / f"{demo_name}.json"
                    #     # CHANGE TO INCLUDE DEMO NUMBER
                    # with open(output_path, "w", encoding="utf-8") as f:
                    #     # The response object, when response_schema is used, contains the Pydantic model directly in its text/json attribute or can be accessed via .to_dict()
                    #     # For structured responses using response_schema, the content is typically directly accessible.
                    #     # If response.json() or response.to_dict() works, use that. Otherwise, try response.text and parse it.
                    #     # Given the schema, response.candidates[0].content.parts[0].text or response.to_dict() is likely what you need.
                    #     # The documentation for the Python client for Gemini API suggests using response.text for string content, or response.to_dict() for structured content.
                    #     # Since response_schema is set to Annotation, the response should already conform to that.
                    #     # Let's assume response.json() or response.to_dict() will provide the serializable content.
                    #     # If the Pydantic model is directly returned as part of the response object, you might need to convert it.
                    #     # The safest bet is often to access the raw text and then parse it if it's JSON string, or use .to_dict()
                    #     json.dump(data, f, indent=4, ensure_ascii=False)
                    
                    # os.remove(pdf_filename)

                    
                    # # # output_path = Path(VIDEOS_PATH) / task.name / f"{annotation_name}.json"
                    # # with open(output_path, "w", encoding="utf-8") as f:
                    # #     json.dump(response, f, ensure_ascii=False, indent=4)
                    # # # Use the response as a JSON string.
