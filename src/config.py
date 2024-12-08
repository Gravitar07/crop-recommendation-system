import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
print(BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, 'final_models')

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

GROQ_MODEL_NAME = 'llama3-8b-8192'