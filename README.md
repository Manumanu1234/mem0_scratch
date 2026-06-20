# mem0

**Conversational AI with persistent long-term memory.** mem0 is a CLI-based chatbot that remembers facts about users across sessions using vector search and an autonomous memory management agent.

Built with [DSPy](https://github.com/stanfordnlp/dspy) for LLM orchestration, [Qdrant](https://qdrant.tech/) for vector storage, [Google Gemini](https://ai.google.dev/) for embeddings and memory extraction, and [Groq](https://groq.com/) for chat inference.

## Features

- **Persistent long-term memory** — User facts extracted, embedded, and stored per-user in Qdrant, persisted across sessions.
- **Autonomous memory management** — DSPy ReAct agent decides on each turn whether to ADD, UPDATE, DELETE, or NOOP memories based on conversation context.
- **Context-aware responses** — The chat LLM retrieves relevant memories before answering, providing continuity across conversations.
- **Multi-user support** — All operations scoped by `user_id`.
- **Faceted search** — Memories filterable by category via Qdrant facets.
- **Interactive CLI** — Simple chat loop in `main.py`.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  main.py     │────▶│  DSPy ReAct      │────▶│  Groq LLM       │
│  (chat loop) │     │  (agent + tools)  │     │  (llama-3.3-70b)│
└──────────────┘     └──────────────────┘     └─────────────────┘
                           │
                    ┌──────┴──────┐
                    │  Memory     │
                    │  Agent      │
                    │  (add/upd/  │
                    │   del/noop) │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │  Extraction & Embedding  │
              │  (Gemini 2.5 Flash +     │
              │   gemini-embedding-2)    │
              └────────────┬────────────┘
                           │
                    ┌──────┴──────┐
                    │  Qdrant DB  │
                    │  (vector DB)│
                    └─────────────┘
```

## Prerequisites

- Python >= 3.12
- A running [Qdrant](https://qdrant.tech/) instance (default: `http://localhost:6333`)
- API keys for [Google Gemini](https://ai.google.dev/) and [Groq](https://groq.com/)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd mem0

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install the package and dependencies
pip install -e .
# or if using uv:
uv sync
```

## Configuration

Create a `.env` file in the project root:

```env
GEMINI_API="your_gemini_api_key"
GROQ_API="your_groq_api_key"
```

Configuration is managed in `settings/config.py`:

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `api_key` | `GEMINI_API` | — | Gemini API key |
| `groq_api` | `GROQ_API` | — | Groq API key |
| `extraction_model` | — | `gemini/gemini-2.5-flash` | Model for memory extraction |
| `embedding_model` | — | `gemini-embedding-2-preview` | Model for embeddings |
| `groqllm` | — | `groq/openai/gpt-oss-120b` | Chat model |
| `qdranturl` | — | `http://localhost:6333` | Qdrant server URL |
| `collection_name` | — | `memories` | Qdrant collection name |

## Usage

### Start the interactive chat

```bash
python main.py
```

The AI retrieves relevant memories before responding and automatically updates its memory after each turn.

```
Let's begin to chat!
User: My name is Jean and I work as a software engineer.
AI: Nice to meet you, Jean! You work as a software engineer.
Trying to update memory...
Memory: 'My name is Jean' was added to DB

User: What is my name?
AI: Your name is Jean.
```

### Initialize Qdrant collection

```bash
python -m services.qdrantDb
```

Creates the `memories` collection with 64-dimensional cosine vectors and payload indexes on `user_id` and `categories`.

### Run manual test

```bash
python testing/test.py
```

## Project Structure

```
mem0/
├── main.py                    # Entry point — interactive chat loop
├── pyproject.toml             # Project metadata and dependencies
├── uv.lock                    # Locked dependency versions
├── .env                       # API keys (not tracked by git)
├── .python-version            # Python 3.12
│
├── agent/
│   └── agent.py               # DSPy ReAct agent for memory management
│
├── schemas/
│   └── pydanticSchema.py      # Pydantic models and DSPy signatures
│
├── services/
│   ├── embedding.py           # Google Gemini embedding service
│   ├── extraction_info.py     # Memory extraction and embedding pipeline
│   └── qdrantDb.py            # Qdrant vector DB CRUD operations
│
├── settings/
│   └── config.py              # Configuration (API keys, model names, Qdrant URL)
│
└── testing/
    └── test.py                # Manual integration test
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `dspy` | >=3.2.1 | LLM orchestration (signatures, ReAct agents) |
| `qdrant-client` | >=1.18.0 | Qdrant vector database client |
| `dotenv` | >=0.9.9 | Environment variable loading |
| `litellm` | >=1.89.2 | Unified LLM interface |
| `mem0ai[nlp]` | >=2.0.7 | NLP utilities |
| `langchain-google-genai` | *(implicit)* | Google Generative AI embeddings |
| `pydantic` | *(implicit)* | Data validation |

## License

MIT
