from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, set_seed, AutoModelForCausalLM
import torch
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from Evaluation.Hybrid_search import HybridSearch
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import os
from typing import Tuple

load_dotenv()

data_path = '/Users/anureddy/Desktop/SEM03/Natural_Language_Processing_with_Transformers/Project_QA/QA_PROJ/data/papers_latest.json'
openai_api_key = os.getenv("OPENAI_API_KEY")
faiss_db_path = '/Users/anureddy/Desktop/SEM03/Natural_Language_Processing_with_Transformers/Project_QA/_Q-A-INLPT-WS2023/DB/faiss_index'


def get_relevant_context(question: str) -> str:
    """
    Retrieve relevant context for a given question.

    Parameters:
        question (str): The user's question.

    Returns:
        str: Relevant context for answering the question.
    """
    docs = hybrid_search_setup.get_relevant_documents(question)
    context = " ".join([doc.get('page_content', '') for doc in docs[:2]])
    return context


def get_documents_and_metadata(question: str) -> Tuple[str, str]:
    """
    Retrieve documents and metadata relevant to a given question.

    Parameters:
        question (str): The user's question.

    Returns:
        Tuple[str, str]: A tuple containing context and metadata string.
    """
    documents = hybrid_search_setup.get_relevant_documents(question)
    context = "\n".join([doc.get('abstract', '') for doc in documents[:2]])
    metadata_str = "\n\n".join([
        f"Title: {doc['title']}\nKeywords: {', '.join(doc['keywords'])}"
        for doc in documents[:2]])
    return context, metadata_str


def prepare_prompt(question: str, context: str, metadata_str: str, answer: str = None) -> str:
    """
    Prepare a prompt for the language model by including context, metadata, and the question.

    Parameters:
        question (str): The user's question.
        context (str): Relevant context for answering the question.
        metadata_str (str): Metadata string related to the documents.
        answer (str, optional): The expected answer. Defaults to None.

    Returns:
        str: The prepared prompt.
    """
    prompt = f"""
    You are a helpful AI Assistant that follows instructions extremely well.
    Use the following context and metadata to answer the user question.
    Think step by step before answering the question. You will get a $100 tip if you provide the correct answer.
    If the answer is not from the context, then print "I don't know." as the response. Don't make your own answers.
    Please ensure your answer is complete and ends with a full-stop.
    QUESTION:
    {question}

    CONTEXT:
    {context}
    METADATA:
    {metadata_str}
    """
    return prompt.strip()


def method_1_pubmed_bert(question: str) -> None:
    """
    Method 1: Using PubMedBERT for Question Answering.

    Parameters:
        question (str): The user's question.

    Returns:
        None
    """
    model_id = "gerardozq/biobert_v1.1_pubmed-finetuned-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer, max_seq_len=1024)
    context, metadata_str = get_documents_and_metadata(question)
    response = qa_pipeline(question=question, context=context)
    if response['score'] < 0.05:
        response['answer'] = 'I dont know'
    print(response)


def method_2_openai_llm(question: str) -> None:
    """
    Method 2: Standard LLM OpenAI.

    Parameters:
        question (str): The user's question.

    Returns:
        None
    """
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=hybrid_search_setup.ensemble_retriever, return_source_documents=True)
    llm_response = qa_chain(question)
    context = ''
    metadatas = []
    for doc in llm_response['source_documents']:
        page_content = doc.page_content
        metadata = metadatas.append(doc.metadata)
        context += page_content
    combined_metadata = "\n".join([f"Title: {m.get('title', '')}\nKeywords: {', '.join(m.get('keywords', []))}" for m in metadatas])
    if "I'm sorry" in llm_response["result"] or "there is no mention" in llm_response["result"] or "I don't know" in llm_response["result"] or "I do not" in llm_response["result"]:
        llm_response["source_documents"] = ""
        llm_response["result"] = "I don't know"
        prompt = prepare_prompt(question, context, combined_metadata)
        print(f"The generated solution: {llm_response}")
    else:
        prompt = prepare_prompt(question, context, combined_metadata)
        print(prompt)
        print(f"The generated response: {llm_response['result']}")


def response_to_dict_v2_v2(generated_response: str) -> dict:
    """
    Convert a generated response to a dictionary.

    Parameters:
        generated_response (str): The generated response.

    Returns:
        dict: A dictionary containing sections such as QUESTION, CONTEXT, METADATA, and ANSWER.
    """
    response_dict = {}

    section_labels = ["QUESTION:", "CONTEXT:", "METADATA:", "ANSWER:"]

    found_labels = []

    for label in section_labels:
        index = generated_response.find(label)
        if index != -1:
            found_labels.append((label, index))

    found_labels.sort(key=lambda x: x[1])

    for i, (label, index) in enumerate(found_labels):
        start_index = index + len(label)
        end_index = None if i + 1 == len(found_labels) else found_labels[i + 1][1]
        content = generated_response[start_index:end_index].strip() if end_index is not None else generated_response[start_index:].strip()
        response_dict[label.strip(":")] = content

    for section in section_labels:
        section_key = section.strip(":")
        if section_key not in response_dict:
            response_dict[section_key] = ""

    return response_dict


def method_3_huggingface_hub(question: str) -> None:
    """
    Method 3: Using HuggingFaceHub.

    Parameters:
        question (str): The user's question.

    Returns:
        None
    """
    model_id = 'HuggingFaceH4/zephyr-7b-beta'
    llm = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature": 0.3, "max_new_tokens": 500}, huggingfacehub_api_token=HF_TOKEN)
    context, metadata_str = get_documents_and_metadata(question)
    prompt = prepare_prompt(question, context, metadata_str)
    generated_text = llm.invoke(prompt)
    response_dict = response_to_dict_v2_v2(generated_text)

    if "I don't know" in response_dict["ANSWER"] or not response_dict["ANSWER"].strip():
        response_dict["CONTEXT"] = ""
        response_dict["METADATA"] = ""
        response_dict["ANSWER"] = "I don't know"
    print(response_dict)


def method_4_medalpaca(question: str, model_dir: str) -> None:
    """
    Method 4: Using Medalpaca from Hugging Face as LLMs.

    Parameters:
        question (str): The user's question.
        model_dir (str): Directory to store the model and tokenizer.

    Returns:
        None
    """
    model_id = "medalpaca/medalpaca-7b"
    model_file_path = os.path.join(model_dir, "medalpaca_model")
    tokenizer_file_path = os.path.join(model_dir, "medalpaca_tokenizer")

    if not os.path.exists(model_file_path) or not os.path.exists(tokenizer_file_path):
        print("Downloading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    else:
        print("Loading model and tokenizer from local directory...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path)
        model = AutoModelForCausalLM.from_pretrained(model_file_path)

    pl_loaded = pipeline("text-generation", model=model, tokenizer=tokenizer)
    context, metadata_str = get_documents_and_metadata(question)
    inputs = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    try:
        prompt = prepare_prompt(question, context, metadata_str)
        response = pl_loaded(inputs, max_length=1024, num_return_sequences=1, do_sample=True, truncation=True)
        print("Generated response:\n", response[0]['generated_text'])
    except Exception as e:
        print(f"An error occurred during text generation: {str(e)}")


if __name__ == "__main__":
    model_dir = "/Users/anureddy/Desktop/SEM03/Natural_Language_Processing_with_Transformers/Project_QA/_Q-A-INLPT-WS2023/Medalpeca"
    os.makedirs(model_dir, exist_ok=True)
    question = 'Are keratin 8 Y54H and G62C mutations associated with inflammatory bowel disease?'
    method_1_pubmed_bert(question)
    # method_2_openai_llm(question)
    # method_3_huggingface_hub(question)
    # method_4_medalpaca(question, model_dir)
