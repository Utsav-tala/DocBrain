from dotenv import load_dotenv
import os

load_dotenv()

# Debug: check if key is loading at all
api_key = os.getenv("OPENAI_API_KEY")
print("Key loaded:", api_key[:10] if api_key else "NOT FOUND — .env not loading")

# Test OpenAI
from openai import OpenAI
client = OpenAI(api_key=api_key)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "say hello"}]
)
print("OpenAI works:", response.choices[0].message.content)

# # Test Gemini (new API)
# import google.genai as genai
# client_gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
# response = client_gemini.models.generate_content(
#     model="gemini-2.0-flash",
#     contents="say hello"
# )
# print("Gemini works:", response.text)

print("Gemini test skipped — using GPT-4o-mini as primary")