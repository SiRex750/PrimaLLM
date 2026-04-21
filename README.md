# PrimaLLM

PrimaLLM is a small LLM memory architecture prototype that:

- extracts subject-verb-object facts from text
- verifies claims against trusted context
- stores verified facts in a JSON wiki layer
- loads verified facts into an L1 cache for live chat use

## Project Files

- `main.py` - main chat orchestrator
- `caveman.py` - knowledge graph extraction
- `verification_layer.py` - claim verification with NLI
- `memory_manager.py` - token-budgeted L1 cache
- `wiki_storage.py` - persistent JSON storage for verified facts
- `karpathy_wiki.json` - local fact store

## Setup

1. Create a local `.env` file in the project root.
2. Add your OpenAI key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

3. Install dependencies inside the virtual environment if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install openai matplotlib networkx tiktoken python-dotenv
```

## Run the Chat App

Start the interactive loop:

```powershell
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe .\main.py
```

Type your message at the `User:` prompt. Type `exit` to quit.

## How It Works

1. `main.py` loads verified facts from `karpathy_wiki.json` into the L1 cache.
2. User input is sent to the chat model.
3. The response is extracted into knowledge triples.
4. Each triple is verified against the current cache context.
5. Verified facts are saved back to JSON and added to the cache.

## Notes

- The repository contains a knowledge-graph image output at `knowledge_graph.png` from earlier runs.
