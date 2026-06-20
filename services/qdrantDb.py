from datetime import datetime
from typing import Optional,List
from litellm import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client.grpc import ScoredPoint
from qdrant_client.models import PointStruct
from qdrant_client.models import Distance,Filter,VectorParams,models
import asyncio
from settings.config import init
from uuid_utils import uuid4
from schemas.pydanticSchema import EmbeddedMemory, RetrievedMemory
import numpy as np


class QdrantOperations:
    def __init__(self,url:str=init.qdranturl,collection_name:str=init.collection_name):
        self.url = url
        self.collection_name = collection_name
    
    @staticmethod
    def convert_retrieved_records(point) -> RetrievedMemory:
        return RetrievedMemory(
            point_id=point.id,
            user_id=point.payload["user_id"],
            memory_text=point.payload["memory_text"],
            categories=point.payload["categories"],
            date=point.payload["date"],
            score=point.score,
        )
    
    async def create_memory_collection(self,vector_size:int=64):
        client = AsyncQdrantClient(url=self.url)
        if not (await client.collection_exists(collection_name=self.collection_name)):
            await client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

            await client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema=models.PayloadSchemaType.INTEGER
            )
            
            await client.create_payload_index(
                collection_name=self.collection_name,
                field_name="categories",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            print(f"Collection '{self.collection_name}' created successfully.")
        else:
            print(f"Collection '{self.collection_name}' already exists.")
    
    async def insert_memories(self,memories:List[EmbeddedMemory]):
        client = AsyncQdrantClient(url=self.url)
        points = []
        for mem in memories:
            point = PointStruct(
                id=uuid4().hex,
                vector=mem.embedding,
                payload={
                    "user_id": mem.user_id,
                    "memory_text": mem.memory_text,
                    "categories": mem.categories,
                    "date": mem.date
                }
            )
            points.append(point)
        await client.upsert(collection_name=self.collection_name, points=points)
        print(f"Inserted {len(points)} memories into collection '{self.collection_name}'.")
    
    async def search_memories(
            self,
            query_embedding: List[float],
            user_id: int,
            top_k: int = 5,
            category_filter: Optional[List[str]] = None,
        ) -> List[RetrievedMemory]:
        
        client= AsyncQdrantClient(url=self.url)
        must_conditions: List[models.FieldCondition] = [
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id) 
            )
        ]
        if category_filter:
            must_conditions.append(
                models.FieldCondition(
                    key="categories",
                    match=models.MatchAny(any=category_filter)
                )
            )
        res= await client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            with_payload=True,
            query_filter=Filter(must=must_conditions),
            score_threshold=0.1,
            limit=2
        )
        
        return [
            self.convert_retrieved_records(points) for points in res.points if points is not None
        ]
        
    async def delete_user_records(self,user_id):
        client=AsyncQdrantClient(url=self.url)
        await client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id", match=models.MatchValue(value=user_id)
                        )
                    ]
                )
            ),
        )


    async def delete_records(self,point_ids):
        client=AsyncQdrantClient(url=self.url)
        await client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=point_ids),
        )


    async def fetch_all_user_records(self,user_id):
        client=AsyncQdrantClient(url=self.url)
        out = await client.query_points(
            collection_name=self.collection_name,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id", match=models.MatchValue(value=user_id)
                    )
                ]
            ),
        )
        return [self.convert_retrieved_records(point) for point in out.points]


    async def get_all_categories(self,user_id):
        client=AsyncQdrantClient(url=self.url)
        COLLECTION_NAME=self.collection_name
        facet_filter = Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
            ]
        )
        
        facet_result = await client.facet(
            collection_name=COLLECTION_NAME,
            key="categories",
            facet_filter=facet_filter,
            limit=1000,  
        )
        facet_result = await client.facet(
            collection_name=COLLECTION_NAME,
            key="categories",
            facet_filter=facet_filter,
            limit=1000, 
        )
        unique_categories = [hit.value for hit in facet_result.hits]

        return unique_categories


    def stringify_retrieved_point(self,retrieved_memory: RetrievedMemory):
        return f"""{retrieved_memory.memory_text} (Categories: {retrieved_memory.categories}) Relevance: {retrieved_memory.score:.2f}"""


if __name__ == "__main__":
    obj=QdrantOperations()
    asyncio.run(obj.create_memory_collection())