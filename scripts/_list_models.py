import os
from dotenv import load_dotenv
load_dotenv(override=True)
from google import genai
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
for m in client.models.list():
    name = m.name.lower()
    if 'flash' in name or ('pro' in name and 'vision' not in name):
        print(m.name)
