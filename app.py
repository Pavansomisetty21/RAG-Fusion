
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from pprint import pprint 
from langchain_groq import ChatGroq 
llm=ChatGroq(api_key="your api key ",model="llama-3.1-8b-instant")

loader = JSONLoader(
    file_path="your jsonfile"
    jq_schema='.[]',  # root-level array
    text_content=False,
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)


class CustomSentenceTransformerEmbedder(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()  # Convert NumPy array to list of lists

    def embed_query(self, text):
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()  # Convert NumPy vector to list


embedding = CustomSentenceTransformerEmbedder()

vectorstore = FAISS.from_documents(splits, embedding)

retriever = vectorstore.as_retriever()


## RAG fusion
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (prompt_rag_fusion |llm    | StrOutputParser() | (lambda x: x.split("\n")))

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results

question = "what is an ai agent"
retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
docs = retrieval_chain_rag_fusion.invoke({"question": question})
pprint(docs)

len(docs)
