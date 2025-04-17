# Install required libraries
# pip install langchain sentence-transformers chromadb

# Import libraries
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# Load and process the text
loader = TextLoader("E:\Projects\Meditations\meditations.txt",encoding="utf-8")
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(documents)

# Create embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(splits, embedding_model)

# Function to retrieve relevant excerpts
def get_relevant_excerpts(query, top_k=3):
    results = vector_store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]

# Example usage
excerpts = get_relevant_excerpts("How to deal with sadness?")
for i, excerpt in enumerate(excerpts):
    print(f"Excerpt {i+1}:\n{excerpt}\n")
