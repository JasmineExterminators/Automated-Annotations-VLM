from google import genai
client = genai.Client(api_key="AIzaSyDjnJusDy6ZyKhylNP-qot_ZgRSJOaoepo") 
MODEL = "gemini-2.5-flash-preview-05-20"

print("making API call...")
scene_of_a_duck_in_the_water_with_a_fish_swimming_nearby = client.files.upload(file="data/duck_in_the_water_eating_a_fish.webp")
response = client.models.generate_content(
                                        model=MODEL, 
                                        contents=[scene_of_a_duck_in_the_water_with_a_fish_swimming_nearby, "what is the name of the file? ducks?"], # TODO CHANGE THIS LINE
                                    )

print(response.text)  # Print the generated content