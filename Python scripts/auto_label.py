from google import genai
client = genai.Client(api_key="YOUR_API_KEY")
api_key = "AIzaSyA3JNDI0RArQ2V7X_eF_P6Y3DX8gP5hGDQ";

# 1. Convert video frames to pdf



# 2. Feed pdf, reference annotation, prompt to Gemini



# 3. save


response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works in a few words"
)
print(response.text)
