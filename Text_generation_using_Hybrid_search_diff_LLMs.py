# Import necessary packages
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, set_seed,AutoModelForCausalLM
import torch
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from Hybrid_search import HybridSearch
from dotenv import load_dotenv



import os
load_dotenv() 

data_path = '/Users/anureddy/Desktop/SEM03/Natural_Language_Processing_with_Transformers/Project_QA/QA_PROJ/data/papers_latest.json'
openai_api_key = os.environ.get("OPENAI_API_KEY")
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
    docs = hybrid_search_setup.get_relevant_documents(question)
    context = " ".join([doc.get('page_content', '') for doc in docs[:2]])
    return context

def get_documents_and_metadata(question):
    documents = hybrid_search_setup.get_relevant_documents(question)
    # Adjusted to use .get() for safer key access
    context = "\n".join([doc.get('abstract', '') for doc in documents[:2]])
    metadata_str = "\n\n".join([
        f"Title: {doc['title']}\nKeywords: {', '.join(doc['keywords'])}"
        for doc in documents[:2]])
    return context, metadata_str

def prepare_prompt(question, context, metadata_str,answer=None):
    """
    Prepare the prompt by including context, metadata, and the question for the LLM.
    """
    prompt = f"""
    You are a helpful AI Assistant that follows instructions extremely well.
    Use the following context and metadata to answer the user question.

    Think step by step before answering the question. You will get a $100 tip if you provide the correct answer.
    If the answer is not from the context then say "I don't know."
    Please ensure your answer is complete and ends with a full-stop.
    QUESTION:
    {question}

    ANSWER:
    {answer}
    
    CONTEXT:
    {context}

    METADATA:
    {metadata_str}
    """
    return prompt.strip()

# Method 1: Using PubMedBERT for Question Answering
def method_1_pubmed_bert(question):
    model_id = "gerardozq/biobert_v1.1_pubmed-finetuned-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
    context,metadata_str = get_documents_and_metadata(question)
    qa_input = {"question": question, "context": context}
    response = qa_pipeline(qa_input)
    prompt = prepare_prompt(question,context,metadata_str,response['answer'])
    print(prompt)


# Method 2: Standard LLM OpenAI
def method_2_openai_llm(question):
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=hybrid_search_setup.ensemble_retriever, return_source_documents=True)
    llm_response = qa_chain(question)
    #print(llm_response)
    context = ''
    metadatas = []
    for doc in llm_response['source_documents']:
        page_content = doc.page_content
        metadata = metadatas.append(doc.metadata)
        context+=page_content
    combined_metadata = "\n".join([f"Title: {m.get('title', '')}\nKeywords: {', '.join(m.get('keywords', []))}" for m in metadatas])
    prompt = prepare_prompt(question,context,combined_metadata,llm_response['result'])
    print(prompt)

# Method 3: Using HuggingFaceHub
def method_3_huggingface_hub(question):
    model_id = 'HuggingFaceH4/zephyr-7b-beta'
    llm = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature": 0.3, "max_new_tokens": 150}, huggingfacehub_api_token=HF_TOKEN)
    context, metadata_str = get_documents_and_metadata(question)
    prompt = prepare_prompt(question, context, metadata_str)
    generated_text = llm.invoke(prompt)
    print("Generated response:\n", generated_text)

# Method 4: Using Medalpaca from Hugging Face as LLMs
def method_4_medalpaca(question, model_dir):
    model_id = "medalpaca/medalpaca-7b"
    
    # Define paths for the model and tokenizer
    model_file_path = '/Users/anureddy/Desktop/SEM03/Natural_Language_Processing_with_Transformers/Project_QA/_Q-A-INLPT-WS2023/DB/Medalpeca/medalpaca_model'
    tokenizer_file_path = '/Users/anureddy/Desktop/SEM03/Natural_Language_Processing_with_Transformers/Project_QA/_Q-A-INLPT-WS2023/DB/Medalpeca/medalpaca_tokenizer'

    # Check if the model and tokenizer are already downloaded
    if not os.path.exists(model_file_path) or not os.path.exists(tokenizer_file_path):
        print("Downloading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        # Save the model and tokenizer locally
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    else:
        print("Loading model and tokenizer from local directory...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path)
        model = AutoModelForCausalLM.from_pretrained(model_file_path)
    
    # Generate text
    pl_loaded = pipeline("text-generation", model=model, tokenizer=tokenizer)
    context,metadata_str = get_documents_and_metadata(question)
    inputs = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"# Make sure this function is defined
    try:
        
        prompt = prepare_prompt(question,context,metadata_str)
        response = pl_loaded(inputs,max_length=1024,num_return_sequences=1, do_sample=True,truncation=True)
        print("Generated response:\n", response[0]['generated_text'])
    except Exception as e:
        print(f"An error occurred during text generation: {str(e)}")



# Example usage
if __name__ == "__main__":
    model_dir = "/Users/anureddy/Desktop/SEM03/Natural_Language_Processing_with_Transformers/Project_QA/_Q-A-INLPT-WS2023/Medalpeca"
    os.makedirs(model_dir, exist_ok=True)
    question = "What are the benefits of using biopolymer-based films in food packaging?"
    #method_1_pubmed_bert(question)
    #method_2_openai_llm(question)
    #method_3_huggingface_hub(question)
    method_4_medalpaca(question,model_dir)
