# Import necessary packages
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, set_seed, AutoModelForCausalLM
import torch
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from Hybrid_search import HybridSearch
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import os
load_dotenv() 

data_path = '/Users/anureddy/Desktop/SEM03/Natural_Language_Processing_with_Transformers/Project_QA/QA_PROJ/data/papers_latest.json'
openai_api_key = os.getenv("OPENAI_API_KEY")
#print(f"API Key: {os.getenv('OPENAI_API_KEY')}")
faiss_db_path = '/Users/anureddy/Desktop/SEM03/Natural_Language_Processing_with_Transformers/Project_QA/_Q-A-INLPT-WS2023/DB/faiss_index'

# Initialize HybridSearchSetup
hybrid_search_setup = HybridSearch(data_path, faiss_db_path)
documents = hybrid_search_setup.load_data()
docs = hybrid_search_setup.transform_data_to_documents(documents)
bm25_retriever = hybrid_search_setup.initialize_bm25_retriever(docs)
faiss_vectorstore = hybrid_search_setup.process_documents_with_faiss(docs)
hybrid_search_setup.create_ensemble_retriever(bm25_retriever, faiss_vectorstore)

HF_TOKEN = "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"

# Helper functions
def get_relevant_context(question):
    """
    Get relevant context from HybridSearch for a given question.

    Args:
        question (str): User's question.

    Returns:
        str: Relevant context.
    """
    docs = hybrid_search_setup.get_relevant_documents(question)
    context = " ".join([doc.get('page_content', '') for doc in docs[:2]])
    return context

def get_documents_and_metadata(question):
    """
    Get relevant documents and metadata from HybridSearch for a given question.

    Args:
        question (str): User's question.

    Returns:
        tuple: Tuple containing context and metadata string.
    """
    documents = hybrid_search_setup.get_relevant_documents(question)
    # Adjusted to use .get() for safer key access
    context = "\n".join([doc.get('abstract', '') for doc in documents[:2]])
    metadata_str = "\n\n".join([
        f"Title: {doc['title']}\nKeywords: {', '.join(doc['keywords'])}"
        for doc in documents[:2]])
    return context, metadata_str

def prepare_prompt(question, context, metadata_str, answer=None):
    """
    Prepare the prompt by including context, metadata, and the question for the LLM,
    and handle the case when the answer is "I don't know."

    Args:
        question (str): User's question.
        context (str): Relevant context.
        metadata_str (str): Metadata string.
        answer (str): Optional answer.

    Returns:
        str: Prepared prompt.
    """
    prompt = f"""
    You are a helpful AI Assistant that follows instructions extremely well.
    Use the following context and metadata to answer the user question.

    Think step by step before answering the question. You will get a $100 tip if you provide the correct answer.
    If the answer is not from the context then print "I don't know." as the response.Don't make your own answers
    Please ensure your answer is complete and ends with a full-stop.
    QUESTION:
    {question}
    
    CONTEXT:
    {context}

    METADATA:
    {metadata_str}
    """
    return prompt.strip()


# Method 1: Using PubMedBERT for Question Answering
def method_1_pubmed_bert(question):
    """
    Method 1: Using PubMedBERT for Question Answering.

    Args:
        question (str): User's question.
    """
    model_id = "gerardozq/biobert_v1.1_pubmed-finetuned-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer, max_seq_len=1024)
    context, metadata_str = get_documents_and_metadata(question)
    response = qa_pipeline(question=question, context=context)
    #print(response)
    if response['score'] < 0.05:
        response['answer'] = 'I dont know'
    print(response)  
    #prompt = prepare_prompt(question, context, metadata_str)
