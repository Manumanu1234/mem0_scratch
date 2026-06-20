from typing import Dict, List 
from datetime import datetime
from pydantic import BaseModel
from services.embedding import Embeddings
from schemas.pydanticSchema import Memory, MemoryExtract,EmbeddedMemory
from settings.config import init
from services.qdrantDb import QdrantOperations
import dspy
import asyncio
import json


class ExtractionInfo(Embeddings):
    def __init__(self):
        super().__init__()
        self.user_id=1
        self.memory_extractor = dspy.Predict(MemoryExtract)
        self.dboperations=QdrantOperations()
    async def extract_memories_from_messages(self, messages, existing_categories):
        transcript = json.dumps(messages)
        with dspy.context(lm=dspy.LM(model=init.extraction_model, api_key=init.api_key)):
            out = await self.memory_extractor.acall(transcript=transcript)
        return out.memories
    async def embed_memories(self, memories: List[Memory]):
        memory_text = [memory.information for memory in memories]
        embeddings = await self.generate_embeddings(memory_text)
        return embeddings
    async def extract_and_embed(self,messages, existing_categories):
        extractor = ExtractionInfo()
        extracted_memories = await extractor.extract_memories_from_messages(messages, existing_categories)
        embeddings = await extractor.embed_memories(extracted_memories)
        print("Extracted Memories:")
        print("==============")
        for mem in extracted_memories:
            print(f"Information: {mem.information}, Predicted Categories: {mem.predicted_categories}")
        print("==============")
        print("Embeddings:")
        print(len(embeddings), len(embeddings[0]))
        return extracted_memories, embeddings
    
    #testing
    
    async def fill_up_context(self,messages):
        infos,embeds=await self.extract_and_embed(messages,[])
        current_date=datetime.now().strftime("%Y-%m-%d %H:%m")
        memories=[
            EmbeddedMemory(
                user_id=self.user_id,
                memory_text=infos.information,
                categories=infos.predicted_categories,
                date=current_date,
                embedding=embeds
            )
            for infos,embeds in zip(infos,embeds)
        ]
        print("\n".join([f"{mem.memory_text} ({mem.categories})" for mem in memories]))
        await self.dboperations.insert_memories(memories)
        print("Memories Inserted Sucessfully\n")
        return {"result":"Memories inserted Sucessfuly"}
    async def run_test_question(self, question):
        question_vector = (await self.generate_embeddings([question]))[0]

        retrieved_memories = await self.dboperations.search_memories(
            query_embedding=question_vector,
            user_id=self.user_id
        )

        memory_str = "\n- ".join([
            self.dboperations.stringify_retrieved_point(r)
            for r in retrieved_memories
        ])

        print(f"*** Question: {question}\nContext:\n{memory_str}\n***")
        return {"result":"Fetched From DB"}
