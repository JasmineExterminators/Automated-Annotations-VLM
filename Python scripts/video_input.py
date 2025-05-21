from google import genai
from google.genai import types


client = genai.Client(api_key="AIzaSyA3JNDI0RArQ2V7X_eF_P6Y3DX8gP5hGDQ")

VIDEO_PATH = "C:/Users/cajas/Downloads/libero_100/libero_90_videos/STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf_demo/demo_49.mp4"

# Only for videos of size <20Mb
video_bytes = open(VIDEO_PATH, 'rb').read()

response = client.models.generate_content(
    model='models/gemini-2.0-flash',
    contents=types.Content(
        parts=[
            types.Part(
                inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
            ),
            types.Part(text=''' This is a video of the robot completing the following goal: to pick up the book and place it in the right compartment of the caddy
                                The left side of the video shows the front view and the right side shows the view on the claw of the robot

                                Can you segment this task with action commands and its reasoning (written in first person), and give the start and end times (exact to one tenth of a second) of an action? 
                            ''')
        ]
    )
)

print(response.text)


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