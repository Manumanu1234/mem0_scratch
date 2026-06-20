import os
from dotenv import load_dotenv
load_dotenv()


class Config:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API")
        self.groq_api= os.getenv("GROQ_API")
        self.extraction_model = "gemini/gemini-2.5-flash"
        self.embedding_model = "gemini-embedding-2-preview"
        self.groqllm="groq/openai/gpt-oss-120b"
        self.qdranturl="http://localhost:6333"
        self.collection_name="memories"
        
init=Config()