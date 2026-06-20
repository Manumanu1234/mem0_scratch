import warnings

warnings.filterwarnings("ignore")

from services.embedding import Embeddings
from agent.agent import update_memories
from services.qdrantDb import   QdrantOperations
from schemas.pydanticSchema import ResponseGenerator
from settings.config import init
import asyncio
import dspy

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
qobj=QdrantOperations()
eobj=Embeddings()

model =dspy.LM(
            model="groq/llama-3.3-70b-versatile",  
            api_key=init.groq_api,
            temperature=1.0                   
        )
async def run_chat(user_id):

    async def fetch_similar_memories(search_text: str, categories: list[str]):
        """
        Search memories from vector database if conversation requires additional context.

        Args:
        - search_text : The string to embed and do vector similarity search
        - categories : List of strings taken from existing_categories. Use an empty list ( [] ) if you want to search across all categories.
        """

        print("Search text: ", search_text)
        print("Categories: ", categories)

        search_vector = (await eobj.generate_embeddings([search_text]))[0]
        memories = await qobj.search_memories(
            query_embedding=search_vector,
            user_id=user_id,
            category_filter=None if len(categories) == 0 else categories,
        )
        memories_str = [qobj.stringify_retrieved_point(m_) for m_ in memories]
        print(f"Retrieved memories: \n", "\n- ".join(memories_str))
        return {"memories": memories_str}

    response_generator = dspy.ReAct(
        ResponseGenerator, tools=[fetch_similar_memories], max_iters=2
    )
    past_messages = []

    existing_categories = await qobj.get_all_categories(user_id=user_id)

    print("Let's begin to chat!")

    while True:
        question = input("User: ")
        with dspy.context(lm=model):
            out = await response_generator.acall(
                    transcript=past_messages, 
                    question=question,
                    existing_categories=existing_categories,
                )

            response = out.response

            past_messages.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response},
                ]
            )

            if out.save_memory:
                print("Trying to update memory...")
                update_result = await update_memories(
                    user_id=user_id,
                    messages=past_messages[-2:],
                )
                print(update_result)
                existing_categories = await qobj.get_all_categories(user_id=user_id)

        print(f"\nAI: {response}\n\n")
    
if __name__ == "__main__":
    asyncio.run(run_chat(user_id=1))