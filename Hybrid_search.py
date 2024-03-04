import json
import os
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from tqdm.auto import tqdm
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

def suppress_warnings():
    """
    A function to suppress DeprecationWarnings.
    """
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    suppress_warnings()

class HybridSearch:
    def __init__(self, data_path):
        """
        Initialize the HybridSearch instance.

        Args:
            data_path (str): Path to the data file.
        """
        self.data_path = data_path
        os.environ['OPENAI_API_KEY'] = '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'
        self.embedding = OpenAIEmbeddings()
        self.ensemble_retriever = None

    def load_data(self):
        """
        Load data from the specified data path.

        Returns:
            dict: Loaded data.
        """
        with open(self.data_path, 'r') as file:
            data = json.load(file)
        return data

    def initialize_bm25_retriever(self, docs):
        """
        Initialize and configure the BM25 retriever.

        Args:
            docs (list): List of Document instances.

        Returns:
            BM25Retriever: Initialized BM25Retriever instance.
        """
        if not all(isinstance(doc, Document) for doc in docs):
            raise ValueError("All items in docs must be Document instances.")
        abstracts = [doc.page_content for doc in docs]
        bm25_retriever = BM25Retriever.from_texts(abstracts, metadatas=[doc.metadata for doc in docs])
        bm25_retriever.k = 3
        return bm25_retriever

    def transform_data_to_documents(self, data):
        """
        Transform raw data into Document instances.

        Args:
            data (dict): Raw data.

        Returns:
            list: List of Document instances.
        """
        docs = []
        for doc in data:
            title = doc.get('title', {}).get('full_text', '')
            abstract = doc.get('abstract', {}).get('full_text', '')
            keywords = doc.get('keywords', [[]])[0] if doc['keywords'] and isinstance(doc['keywords'][0], list) else []
            document = Document(page_content=abstract, metadata={'title': title, 'keywords': keywords})
            docs.append(document)
        return docs

    def process_documents_with_chroma(self, docs):
        """
        Process documents with Chroma vector store.

        Args:
            docs (list): List of Document instances.

        Returns:
            Chroma: Chroma vector store.
        """
        persist_directory = './chroma_openai'
        db3 = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
        if not all(isinstance(doc, Document) for doc in docs):
            raise ValueError("All items in docs must be Document instances.")
        chroma_retriever = db3.as_retriever(search_kwargs={'k': 3})
        return chroma_retriever

    def create_ensemble_retriever(self, bm25_retriever, chroma_retriever):
        """
        Create an ensemble retriever.

        Args:
            bm25_retriever (BM25Retriever): BM25Retriever instance.
            chroma_retriever (Chroma): Chroma vector store.
        """
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.7, 0.3])
        self.ensemble_retriever = ensemble_retriever

    def get_relevant_documents(self, query):
        """
        Get relevant documents based on the user query.

        Args:
            query (str): User's query.

        Returns:
            list: Formatted relevant documents.
        """
        results = self.ensemble_retriever.get_relevant_documents(query)
       
        formatted_results = []
        for document in results:
            doc_info = {
                'title': document.metadata.get('title', document.metadata.get('source/title','No Title')),
                'keywords': document.metadata.get('keywords', []),
                'abstract': document.page_content
            }
            formatted_results.append(doc_info)
        return formatted_results

# Example Usage:
# hs = HybridSearch('./papers_latest.json')
# data = hs.load_data()
# docs = hs.transform_data_to_documents(data)
# bm25_retriever = hs.initialize_bm25_retriever(docs)
# chroma_vectorstore = hs.process_documents_with_chroma(docs)
# hs.create_ensemble_retriever(bm25_retriever, chroma_vectorstore)
