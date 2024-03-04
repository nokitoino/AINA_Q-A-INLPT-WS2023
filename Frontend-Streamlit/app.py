import json
import random
import openai
import streamlit as st
from langchain import hub
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from Evaluation.Hybrid_search import HybridSearch
import warnings

warnings.filterwarnings("ignore")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

st.set_page_config(page_title="Conversational Agent", layout="wide")

@st.experimental_singleton
def initialize_components():
    """
    Initialize global components only once.

    Returns:
        Tuple: Tuple containing HybridSearch instance, RAG prompt, ChatOpenAI instance, and ContextualCompressionRetriever instance.
    """
    data_path = './papers_latest.json'
    hs = HybridSearch(data_path)
    data = hs.load_data()
    docs = hs.transform_data_to_documents(data)
    bm25_retriever = hs.initialize_bm25_retriever(docs)
    chroma_vectorstore = hs.process_documents_with_chroma(docs)
    hs.create_ensemble_retriever(bm25_retriever, chroma_vectorstore)
    
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    compressor = LLMChainExtractor.from_llm(llm=llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=hs.ensemble_retriever
    )
    
    return hs, prompt, llm, compression_retriever

def display_conversation_interface(compression_retriever, prompt, llm):
    """
    Display the main conversation interface.

    Args:
        compression_retriever (ContextualCompressionRetriever): Instance of the compression retriever.
        prompt (str): RAG prompt.
        llm (ChatOpenAI): Instance of ChatOpenAI model.
    """
    st.title("Your PubMed Bot")

    if 'query' not in st.session_state:
        st.session_state['query'] = ""
    
    query = st.text_input("Ask me anything about PubMed:", value=st.session_state['query'], key="query_input")

    if st.button('Ask') or query:
        st.session_state['query'] = query
        generate_response_and_follow_ups(query, compression_retriever, prompt, llm)

def generate_response_and_follow_ups(query, compression_retriever, prompt, llm):
    """
    Generate response and follow-up questions based on the user query.

    Args:
        query (str): User's query.
        compression_retriever (ContextualCompressionRetriever): Instance of the compression retriever.
        prompt (str): RAG prompt.
        llm (ChatOpenAI): Instance of ChatOpenAI model.
    """
    compressed_docs = compression_retriever.get_relevant_documents(query)
    rag_chain_compressor = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    generated_response = rag_chain_compressor.invoke(query)
    st.write("Response:", generated_response)
    st.divider()
    titles_and_keywords = []
    st.subheader('Context: \n')
    for document in compressed_docs:
        st.write(document.page_content)
        title = document.metadata.get('title', 'No Title')
        keywords = document.metadata.get('keywords', [])
        titles_and_keywords.append({'title': title, 'keywords': keywords})
    st.subheader('Related Documents: \n')    
    for item in titles_and_keywords:
        if not item['title'] == 'No Title':
            st.write(f"Title: \n{item['title']}")
            st.write(f"Keywords: \n{', '.join(item['keywords'])}")
            st.write("\n")  
            st.divider()
        
    follow_up_questions = generate_questions(compressed_docs, 3)
    questions_list = json.loads(follow_up_questions)
    for q in questions_list.values():
        if st.button(q):
            st.session_state['query'] = q
            st.experimental_rerun()

def format_docs(docs):
    """
    Format documents for display.

    Args:
        docs (List): List of documents.

    Returns:
        str: Formatted document content.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def generate_questions(context, amount):
    """
    Generate follow-up questions based on the context.

    Args:
        context (str): Context for generating questions.
        amount (int): Number of questions to generate.

    Returns:
        str: JSON formatted follow-up questions.
    """
    question_type = ["confirmation questions [yes or no]", "factoid-type questions [what, which, when, who, how]",
                     "list-type questions", "casual questions [why or how]", "hypothetical questions [e.g. what would happen if...]", "questions"]
    random_type = random.randint(0, len(question_type) - 1)

    prompt = f"""Context: {context}\n\n. Form {question_type[random_type]} in your own words. Answer in this json format {{"question_1": '', ...}} (Exact Amount: {amount}), don't say anything else."""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "assistant"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=500,
        n=2
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    hs, prompt, llm, compression_retriever = initialize_components()
    display_conversation_interface(compression_retriever, prompt, llm)
