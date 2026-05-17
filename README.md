AI SOUS CHEF (rag + llm cooking assistant)

this is a rag-powered cooking assistant that combines a local llm with a custom knowledge base to answer cooking questions, suggest recipes, and help you figure out what to make with what you have. the main point wasn't just to build a chatbot — it was to understand how rag systems actually work: how documents get chunked, embedded, retrieved, and passed as context to an llm so it stays grounded in real information instead of just guessing.

DATA

the assistant's knowledge base is built from three sources:

- recipenlg dataset (first 5000 recipes) — titles, ingredients, and directions
- cooking tip txt files — general technique and kitchen advice
- ingredient substitution txt files — what to use when you're missing something

everything gets chunked, embedded, and stored in a local faiss vector index during ingestion. the dataset was capped at 5000 recipes to keep ingestion fast while still giving the model enough variety to be useful.

FEATURES

for each query, the system:

- embeds the user's question using sentence-transformers
- retrieves the most relevant chunks from the faiss vectorstore
- passes those chunks as context to the llm
- the llm uses that context to give a grounded, specific answer

the retrieval step is what keeps answers accurate — without it, the llm would just rely on general training knowledge. with it, answers are tied to actual recipes and tips from the knowledge base.

ARCHITECTURE

- embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, no api needed)
- vector store: faiss (local index, saved to disk)
- llm: ollama (runs locally)
- framework: langchain
- interface: flask web app

the whole thing runs offline once set up. no openai api, no cloud calls.

KNOWLEDGE BASE

- ~5000 recipes with ingredients and step-by-step directions
- cooking tips covering technique, timing, temperature, etc
- substitution guides for common ingredients (eggs, butter, flour, dairy, etc)

CHUNKING

- chunk size: 500 characters
- chunk overlap: 50 characters
- splitter: recursive character text splitter

HOW IT WORKS

1. run ingest.py to build the vector index from your data files
2. start the flask app
3. ask it anything — "what can i make with chicken and lemon?", "how do i substitute buttermilk?", "what's the difference between sautéing and pan-frying?"

the llm handles the conversation and reasoning. the rag pipeline keeps it grounded in the actual documents you gave it.

LIMITATIONS

- only knows what's in the knowledge base — limited to the 5000 recipes ingested
- no memory between sessions (each conversation starts fresh)
- retrieval quality depends on how well the query matches the indexed chunks
- local llm quality depends on which ollama model you're running

NEXT STEPS

- add more recipes and expand the substitution library
- add conversation memory so follow-up questions work better
- experiment with reranking retrieved chunks before passing to llm
- try a larger embedding model for better retrieval accuracy

HOW TO RUN

install dependencies:
pip install -r requirements.txt

build the vector index:
python ingest.py

start the app:
python app.py