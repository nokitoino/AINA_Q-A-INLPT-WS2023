import json
import os
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.auto import tqdm

class HybridSearch:
    def __init__(self, data_path, faiss_index_path):
        self.data_path = data_path
        self.faiss_index_path = faiss_index_path
        os.environ['OPENAI_API_KEY'] = '<openai api key>'
        self.embedding = OpenAIEmbeddings()
        self.ensemble_retriever = None

    def load_data(self):
        with open(self.data_path, 'r') as file:
            data = json.load(file)
        return data




    def initialize_bm25_retriever(self, docs):
        if not all(isinstance(doc, Document) for doc in docs):
            raise ValueError("All items in docs must be Document instances.")
        abstracts = [doc.page_content for doc in docs]
        bm25_retriever = BM25Retriever.from_texts(abstracts, metadatas=[doc.metadata for doc in docs])
        bm25_retriever.k = 2
        return bm25_retriever
    def transform_data_to_documents(self, data):
        docs = []
        for doc in data:
            title = doc.get('title', {}).get('full_text', '')
            abstract = doc.get('abstract', {}).get('full_text', '')
            keywords = doc.get('keywords', [[]])[0] if doc['keywords'] and isinstance(doc['keywords'][0], list) else []
            document = Document(page_content=abstract, metadata={'title': title, 'keywords': keywords})
            docs.append(document)
        return docs



    def process_documents_with_faiss(self, docs):
        if os.path.exists(self.faiss_index_path):
            print("FAISS index already exists. Loading existing index.")
            return FAISS.load_local(self.faiss_index_path, self.embedding)
        
        # Ensure docs are Document objects
        if not all(isinstance(doc, Document) for doc in docs):
            raise ValueError("All items in docs must be Document instances.")
        abstracts = [doc.page_content for doc in docs]
        metadata = [doc.metadata for doc in docs]
        faiss_vectorstore = None
        batch_size = 1000
        for i in tqdm(range(0, len(abstracts), batch_size), desc="Processing documents"):
            batch_texts = abstracts[i:i+batch_size]
            batch_metadatas = metadata[i:i+batch_size]
            if faiss_vectorstore is None:
                faiss_vectorstore = FAISS.from_texts(batch_texts, self.embedding, batch_metadatas)
            else:
                faiss_vectorstore.update(batch_texts, self.embedding, batch_metadatas)
        
        faiss_vectorstore.save_local(self.faiss_index_path)
        return faiss_vectorstore

    def create_ensemble_retriever(self, bm25_retriever, faiss_vectorstore):
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={'k': 2})
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
        self.ensemble_retriever = ensemble_retriever

    def get_relevant_documents(self, query):
        results = self.ensemble_retriever.get_relevant_documents(query)
        formatted_results = []
        for document in results:
            doc_info = {
                'title': document.metadata.get('title', 'No Title'),
                'keywords': document.metadata.get('keywords', []),
                'abstract': document.page_content
            }
            formatted_results.append(doc_info)
        return formatted_results 

if __name__ == "__main__":
    hs = HybridSearch('<path to papers.json>', 
                      '<path to faiss index>')
    data = hs.load_data()
    docs = hs.transform_data_to_documents(data)
    print(f"First item type before passing to BM25: {type(docs[0])}")
    bm25_retriever = hs.initialize_bm25_retriever(docs)
    
    faiss_vectorstore = hs.process_documents_with_faiss(docs)
    hs.create_ensemble_retriever(bm25_retriever, faiss_vectorstore)

    query = 'Why did the researchers develop text classifiers for detecting invasive fungal diseases from free-text radiology reports?'
    ensemble_results = hs.get_relevant_documents(query)  # This will now receive a list
    for result in ensemble_results:
        print(result)
