from pydantic import BaseModel
from typing import List,Dict
import dspy

class Memory(BaseModel):
    information: str
    predicted_categories: list[str]

class MemoryExtract(dspy.Signature):
    """
    Extract relevant information from the conversation. Create memory entries
    that you should remember when speaking with the user later.

    You will be provided a list of the existing categories in the memory
    database. When predicting the category of this information, you can
    decide to create new categories, or pick from an existing one if it exists.

    Extract informations piece by piece creating atomic units of memory.

    If transcript does not contain any information worth extracting, set
    no_info to True, else False.
    """

    transcript: str = dspy.InputField()
    existing_categories: list[str] = dspy.InputField()
    no_info: bool = dspy.OutputField(
        description="set true if no info to be extracted"
    )
    memories: list[Memory] = dspy.OutputField()


class EmbeddedMemory(BaseModel):
    user_id:int
    memory_text: str
    categories: list[str]
    date: str
    embedding: List[float]

class RetrievedMemory(BaseModel):
    point_id: str
    user_id: int
    memory_text: str
    categories: list[str]
    date: str
    score: float
    
class MemoryWithIds(BaseModel):
    memory_id:int 
    memory_text:str
    
class UpdateMemorySignature(dspy.Signature):
    """
    You will be given the conversation between user and assistant and some similar memories from the database. Your goal is to decide how to combine the new memories into the database with the existing memories.

    Actions meaning:
    - ADD: add new memories into the database as a new memory
    - UPDATE: update an existing memory with richer information.
    - DELETE: remove memory items from the database that aren't required anymore due to new information
    - NOOP: No need to take any action

    If no action is required you can finish.

    Think less and do actions.
    """

    messages: list[dict] = dspy.InputField()
    existing_memories: list[MemoryWithIds] = dspy.InputField()
    summary: str = dspy.OutputField(
        description="Summarize what you did. Very short (less than 10 words)"
    )

class ResponseGenerator(dspy.Signature):
    """
    You will be given a past conversation transcript between user and an AI agent. Also the latest question by the user.

    You have the option to look up the past memories from a vector database to fetch relevant context if required. If you can't find the answer to user's question from transcript or from your own internal knowledge, use the provided search tool calls to search for information.

    You are also provided a list of existing categories in the memory database to quickly search across categories. You can select multiple categories as a list to do your searches. If you select no categories (keep it empty). If you keep categories as empty, we will simply search across the entire database - that is fine too.

    The retrieved information may or may not contain the information user wants. React accordingly.

    You must output the final response, and also decide the latest interaction needs to be recorded into the memory database. New memories are meant to store new information that the user provides.

    While responding, you must be aware that you are continuously learning new memories about the user, so if retrieved memories do not directly address the user's question, mention what you know, acknowledge the gaps in your knowledge, and ask the user for information.

    If you retrieved records using the search tools, and the information was already present, no need to save a new memory. Only save memory if the new information is richer than what you retrieved or didn't find.

    New memories should be made when the USER provides new info. It is not to save information about the the AI or the assistant.
    """

    transcript: list[dict] = dspy.InputField()
    existing_categories: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    response: str = dspy.OutputField()
    save_memory: bool = dspy.OutputField(
        description="True if a new memory record needs to be created for the latest interaction"
    )