
from services.qdrantDb import QdrantOperations,EmbeddedMemory,RetrievedMemory
from services.embedding import Embeddings
from schemas.pydanticSchema import UpdateMemorySignature,MemoryWithIds
from datetime import datetime
from typing import List
from settings.config import init
import dspy
qobj=QdrantOperations()
eobj=Embeddings()
user_id=1


dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

async def update_memories_agent(
    user_id: int, messages: list[dict], existing_memories: list[RetrievedMemory]
):

    def get_point_id_from_memory_id(memory_id):
        return existing_memories[memory_id].point_id

    async def add_memory(memory_text: str, categories: list[str]) -> str:
        """
        Add the new_memory into the database.
        No need to pass any args.
        """
        print("Adding memory: ", memory_text)
        print("Categories: ", categories)

        embeddings = await eobj.generate_embeddings([memory_text])
        await qobj.insert_memories(
            memories=[
                EmbeddedMemory(
                    user_id=user_id,
                    memory_text=memory_text,
                    categories=categories,
                    date=datetime.now().strftime("%Y-%m-%d %H:%m"),
                    embedding=embeddings[0],
                )
            ]
        )

        return f"Memory: '{memory_text}' was added to DB"

    async def update(memory_id: int, updated_memory_text: str, categories: list[str]):
        """
        Updating memory_id to use updated_memory_text

        Args:
        memory_id: integer index of the memory to replace

        updated_memory_text: Simple atomic factoid to replace the old memory with the new memory

        categories: Use existing categories or create new ones if required
        """
        print(
            "Memory updating: ",
            "\n Original: ",
            existing_memories[memory_id].memory_text,
            "\n New memory text: ",
            updated_memory_text,
        )

        point_id = get_point_id_from_memory_id(memory_id)
        await qobj.delete_records([point_id])

        embeddings = await eobj.generate_embeddings([updated_memory_text])

        await qobj.insert_memories(
            memories=[
                EmbeddedMemory(
                    user_id=user_id,
                    memory_text=updated_memory_text,
                    categories=categories,
                    date=datetime.now().strftime("%Y-%m-%d %H:%m"),
                    embedding=embeddings[0],
                )
            ]
        )
        return f"Memory {memory_id} has been updated to: '{updated_memory_text}'"

    async def noop():
        """
        Call this is no action is required
        """
        return "No action done"

    async def delete(memory_ids: list[int]):
        """
        Remove these memory_ids from the database
        """
        print("Deleting these memories")
        for memory_id in memory_ids:
            print(existing_memories[memory_id].memory_text)

        await qobj.delete_records(memory_ids)
        return f"Memory {memory_ids} deleted"

    memory_updater = dspy.ReAct(
        UpdateMemorySignature, tools=[add_memory, update, delete, noop], max_iters=3
    )
    memory_ids = [
        MemoryWithIds(
            memory_id=idx, memory_text=m.memory_text, memory_categories=m.categories
        )
        for idx, m in enumerate(existing_memories)
    ]

    with dspy.context(
        lm = dspy.LM(
            model=init.groqllm,  
            api_key=init.groq_api,
            temperature=1.0                   
        )
    ):
        out = await memory_updater.acall(
            messages=messages, existing_memories=memory_ids
        )
    return out.summary


async def update_memories(user_id: int, messages: list[dict]):
    latest_user_message = [x["content"] for x in messages if x["role"] == "user"][-1]
    embedding = (await eobj.generate_embeddings([latest_user_message]))[0]

    retrieved_memories = await qobj.search_memories(query_embedding=embedding, user_id=user_id)

    response = await update_memories_agent(
        user_id=user_id, existing_memories=retrieved_memories, messages=messages
    )
    return response


async def test():
    messages = [{"role": "user", "content": "I am currently working as Ai engineer"}]
    response = await update_memories(user_id=1, messages=messages)

    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test())