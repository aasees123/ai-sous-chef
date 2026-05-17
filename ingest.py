import os
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

documents = []

# LOAD RECIPE CSV
csv_path = "data/recipes/RecipeNLG_dataset.csv"
df = pd.read_csv(csv_path)

for _, row in df.head(5000).iterrows():
    text = f"""
    Recipe: {row['title']}

    Ingredients:
    {row['ingredients']}

    Directions:
    {row['directions']}
    """

    documents.append(Document(page_content=text))

# LOAD TXT FILES
folders = ["data/cooking_tips", "data/substitutions"]

for folder in folders:
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r") as f:
                text = f.read()

            documents.append(Document(page_content=text))

# SPLIT
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

docs = splitter.split_documents(documents)

# EMBEDDINGS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# VECTOR DB
db = FAISS.from_documents(docs, embeddings)

db.save_local("vectorstore")

print("Vector database created.")
