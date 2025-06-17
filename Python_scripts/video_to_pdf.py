# This script assumes a structure of folder (VIDEO_PATH) -> tasks (folders) -> demos (mp4s)

import cv2
import os
from pathlib import Path
from fpdf import FPDF

VIDEOS_PATH = "C:/Users/wuad3/Downloads/TEST"
 
FRAME_GAP = 10
PHOTO_X = 10
PHOTO_Y = 10
PHOTO_W = 190


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
                            pdf.image(frame_filename, x=PHOTO_X, y=PHOTO_Y, w=PHOTO_W)
                            os.remove(frame_filename)
                        frame_count += 1
                    
                    # When everything done, release the capture
                    cap.release()

                    # Getting PDF
                    destination = os.path.join(task_dir, "pdfs")
                    os.makedirs(destination, exist_ok=True)
                    pdf_path = os.path.join(destination, f"{demo_name}_frames.pdf")
                    
                    pdf.output(pdf_path)
                    
        
