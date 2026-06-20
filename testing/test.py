from services.extraction_info import ExtractionInfo
import asyncio
if __name__=="__main__":
    obj=ExtractionInfo()
    details={
    "context": [
        {
        "role": "user",
        "content": "My name is John and I work as a software engineer."
        },
        {
        "role": "assistant",
        "content": "Nice to meet you, John! A software engineer's perspective is interesting."
        },
        {
        "role": "user",
        "content": "I'm thinking of switching careers to become a chef."
        },
        {
        "role": "assistant",
        "content": "That's a big change! What kind of food are you passionate about?"
        },
        {
        "role": "user",
        "content": "I love Italian food. But my name is not John, it's Jean."
        }
    ],
    "test": [
        {
        "question": "What is my name?",
        "answer": "Your name is Jean."
        },
        {
        "question": "What is my current profession?",
        "answer": "You work as a software engineer."
        },
        {
        "question": "What career are you considering?",
        "answer": "You are considering becoming a chef."
        },
        {
        "question": "What type of food do you love?",
        "answer": "You love Italian food."
        }
    ]
    }
    asyncio.run(obj.run_test_question(
        details['test'][1]['question']
    ))
    
