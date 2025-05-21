import cv2
import os
from dataclasses import dataclass
from typing import List
import re
from datetime import datetime
from ipdb import set_trace
import numpy as np
from utils.annotation import FrameAnnotation, Annotation, parse_annotation_file, render_annotations


FONT_SCALES = {
    "Action": 0.7,
    "Reasoning": 0.5
}

FONT_COLORS = {
    "Action": (50, 255, 50),  # Green
    "Reasoning": (50, 255, 255)  # Yellow
}


if __name__ == "__main__":
    # video_path = f"libero_dataset/libero_90_videos/LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_demo/demo_0.mp4"
    demo_id = 0
    data_path = "libero_dataset/libero_90/LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_demo.hdf5"
    ann_dir = "libero_dataset/libero_90_videos/LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_demo"

    txt_path = ann_dir + f"/demo_{demo_id}.txt"

    # output_path = video_path.replace(".mp4", "_annotated.mp4")
    output_path = ann_dir + f"/demo_{demo_id}_annotated.mp4"

    annotations = parse_annotation_file(txt_path)
    for a in annotations:
        print(f"{a.type}: {a.start_time:.2f}s - {a.end_time:.2f}s -> {a.description}")

    from load_libero_dataset import load_hdf5_to_dict
    data = load_hdf5_to_dict(data_path)["data"]
    video = np.concatenate([
        data[f"demo_{demo_id}"]['obs']['agentview_rgb'][:, ::-1],
        data[f"demo_{demo_id}"]['obs']['eye_in_hand_rgb'][:, :, ::-1],
    ], axis=2)

    render_annotations(video, annotations, output_path)
    print(f"\nAnnotated video saved to: {output_path}\n")