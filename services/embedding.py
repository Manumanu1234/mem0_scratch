from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List
from settings.config import init
gemini_embeddings = GoogleGenerativeAIEmbeddings(model=init.embedding_model,api_key=init.api_key,output_dimensionality=64)

class Embeddings:
    def __init__(self):
        self.embeddings = gemini_embeddings
    async def generate_embeddings(self,texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)
