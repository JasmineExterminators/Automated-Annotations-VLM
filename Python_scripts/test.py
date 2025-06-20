# this is a random test file. ignore

from PIL import Image, ImageDraw
import os
import sys

# Check if folder path is provided
if len(sys.argv) != 2:
    print("Usage: python test.py <folder_path>")
    sys.exit(1)

# Get folder path from command line argument
folder_path = sys.argv[1]

# 1. Create (or ensure) the output folder exists
os.makedirs(folder_path, exist_ok=True)

# 2. Create a blank white canvas
img_size = (200, 200)  # width, height in pixels
img = Image.new('RGB', img_size, 'white')
draw = ImageDraw.Draw(img)

# 3. Draw the yellow face
margin = 10
draw.ellipse(
    (margin, margin, img_size[0] - margin, img_size[1] - margin),
    fill=(255, 255, 0),
    outline='black'
)

# 4. Draw the eyes
eye_radius = 15
left_eye_center  = (60, 70)
right_eye_center = (140, 70)
for cx, cy in (left_eye_center, right_eye_center):
    draw.ellipse(
        (cx - eye_radius, cy - eye_radius, cx + eye_radius, cy + eye_radius),
        fill='black'
    )

# 5. Draw the smile (a simple arc)
smile_box = (60, 80, 140, 150)
draw.arc(smile_box, start=0, end=180, fill='black', width=5)

# 6. Save the image
output_path = os.path.join(folder_path, 'smiley.png')
img.save(output_path)
print(f"Smiley saved to: {output_path}")
