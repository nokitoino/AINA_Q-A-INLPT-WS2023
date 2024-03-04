
This project has been developed as part of the "Natural Language Processing with Transformers" lecture, conducted by Professor Gertz.


Github UserName Mapping

| UserName     | StudentName | Matrikelnummer     |
|----------|-----|----------------|
|   Ibinmbiju   | Ibin Mathew Biju  | 3770662 |
|   AnuR1234  | Anu Reddy | 3768482  |
|   nokitoino   | Burhan Akin Yilmaz | 4114082    |
|   nr59684   | Nilesh Parshotam Rijhwani | 3771253    |

# Medical QA system chatbot (NLP with Transformer Project)

Introduction:

Nowadays, we have opened LLMs (like chatgpt, Bard,Huggingchat…) and the list goes on in order to search for answers to your queries, but most of them are not up-to-date and most importantly, they are not domain-specific. (as in our case, all medical-related information).
Even though they do provide information, the level of precision and accuracy is the point of the question. Most of these LLMs hallucinate most of the answers if they don't know the answer.
Most of the LLMs function as “black boxes” - it’s not easy to understand which sources an LLM was considering when they arrived at their conclusions.
Hence, we thought of introducing (a smart retrieval system similar to RAG, which can be combined with LLM(to generate the text).
By doing so, we are more confident about the answers given now by LLM as they are generated by our own corpus of data.

## Table of Contents
- [High-end Architecture](#high-end-architecture)
- [Documentation](#documentation)
- [Contribution](#contribution)
- [File Overview](#file-overview)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
  
## High-end architecture:
![Unbenanntes_Diagramm drawio](https://github.com/nokitoino/_Q-A-INLPT-WS2023/assets/18616498/74372a75-16e1-4c3c-b87d-c176e97225ba)

There are two phases to the project architecture:
Retrieval of the relevant documents 
Generation of new text from the documents 

Phase 1: Data Retrival
Step 1: Webscrape the content (title, abstract,authors,references) from PubMed website (https://pubmed.ncbi.nlm.nih.gov/?term=intelligence+%5BTitle%2Fabstract%5D&filter=simsearch1.fha&filter=years.2014-2024&sort=date) from 2014-2024 and store it in JSON format.
Step 2: There are 2 options:
1st option : The bulk upload of the json file into Elasticsearch performing customized mappings performing semantic search and providing the relevant documents along with similarity score 
2nd option : Perform chunking using Langchain RecursivecharacterTextSplitter to perform chunking of 1000 with 200 overlap.(used this option in the project)
Step 3: Using the OpenAI embedding API, embed the content and store it in the vector store.
Phase 2: Generation phase 
Step 1: when the user posts a question on UI (steamlit), it is converted into an embedded API 
Step 2: The question posted will be paired up with the vectors in the vector store using semantic search, and the most relevant documents (k value = number of relevant documents) with respect to the questions will be provided to LLM 
Step 3:These documents will be sent in context to the LLM (Medapeca) along with questions in order to generate most suitable answer and displayed in UI with references( most relevant documents).



## Documentation

The documentations can be found in [DOCUMENTATION.md](DOCUMENTATION.md), which contains the following content table.
- [Members Contribution](DOCUMENTATION.md)
- [RAG Pipeline](DOCUMENTATION.md)
- [Experimental Setup](DOCUMENTATION.md)
- [Hardware Requirements](DOCUMENTATION.md)
- [Evaluation](DOCUMENTATION.md)
- [Medical QA-system Demo](DOCUMENTATION.md)
- [Resources](DOCUMENTATION.md)

## Contribution
Teamates contribution(as of now):
1. Anu Reddy:
   - Performed data analysis on the documents from pubmed url link.
   - Performed data retrieval on the combined documents in ndjson format (provided by akin) via writing queries in elastic search in Python( and cross-verified in kibana) and optimized the queries to retrieve the documents using both keyword and vector search on the textual components of the dataset(title and abstract).
   - Performed some test queries by providing the questions as input and retrieving the corresponding documents along with their cosine similarity scores.
   - Experimented with different Embeddings models (openAI,BGE embeddings) and the comparison report has been shared(https://docs.google.com/document/d/1oVKGwl1XahiJP7jK8ojgg4UXNGZ2AUEMDeUFgmyyIZQ/edit?usp=sharing)
   - Worked with Hybrid search (Ensemble Retriver = BM25 retriever+Faiss retriever) (implementation will be uploaded soon)
   - Experimented with usage of different LLMs (openAI,Huggingfacehub(HuggingFaceH4/zephyr-7b-beta) and bloke model Medalpeca(medical LLM) to generate the context (implementation will be uploaded soon)
   - Working on implementation of Hybrid Search
   - Looking to different evaluating metrics 
2. Akin Yilmaz:
   - Developed the PudMedScraper.py using Entrez. Bypassed the Ratelimit of 9999 files using date intervalls. Created the JSON format in cooperation with the others.
   - Testing a parallel Pipeline using Haystack for the entire workflow. Implemented simple pipeline using the DocumentStore using Elasticsearch and experimented with the TF-IDF (sparse) Retriever, and T5 as LLM, aborted the continuation due to the group agreement to stick to LangChain.
   - Implemented and commented embedding_evaluation.py
   - Only Commented and uploaded Embedding-OpenAI-Chroma.ipynb, which was implemented by Anu Reddy.
   - Implemented LLM-GPT3.5-Turbo.ipynb, which is the continuation of our base Notebook Embedding-OpenAI-Chroma.ipynb. It uses GPT3.5 Turbo as LLM.
   - Implemented Evaluation-Contextual-Compression.ipynb based on the ideas used in the lectures last assignment. Uses LLMExtractor, and different metrics to evaluate the performance of our entire pipeline with the help of Hugging Face pubmed_qa dataset.
3. Nilesh Rijhwani:
   - Working on the automation of webscrapping where I am using following structure to maintain timely webscarpping (Droped due to time constraint):
      - Using Python scheduler library - APScheduler
      - Also storing the last scrapped date in a text file amd using functions to access and update the same before putting it in query.
   - Defined a chunk size of 1000 to start working on the experimental phase where the inout json from the webscrapping (papers.json) is brokendown in to defined chunksize and stored as json which later will be used in ambedding.py as an input to our ambedding function
   - Worked on embedding model and function, current decision - 'text_davinci_003' which is gonna be deprectaed in jan 2024, next model --> gpt-3.5-turbo-1106
      - Implemented the Embedding model for PubMed Documents using OpenAI's GPT-3 API.
   - Worked on the pubmedbert LLM for generating the response
   - Performed evaluation of LLMs based on BERT, BLEU and Rogue Score as primary metrics
   - Worked on follow-up questions generation using openai based on compressed context(Developed by Akin)
4. Ibin Mathew Biju:
   - Experimented with different embedding models such as openAI and compared its performance.
   - Researched and worked on different vector stores such as FAISS, chromaDB, pinecone to figure out the best suitable vector store for the architecture.
   - Implemented FAISS vector store and integrated with the current embedding files and did vector search.
   - Worked on different LLMs and experimenting different combinations for better results.
   - Researched about various front end possibilities and hosting services and finally decided to use streamlit for front end and huggingface spaces for free hosting.
   - Developed and Refactored and fine tuned the code for front end
   - Integrated and deployed the model and Designed the UI.


## File Overview, Installation, Usage

The [Scraper](LangChainRAG/) folder contains the document scraper, which is important to execute before using any other scripts from us.

| File     |  Functionality | Requirements
|----------|-----|----------------|
|   PubMedScraper.py  | Creates papers.json   | None

The [LangChain RAG](LangChainRAG/) folder contains base Jupyter Notebooks that our final app is based on. It contains the embedding, vector storage, LLM, Contextual Compression.

| File     |  Functionality | Requirements
|----------|-----|----------------|
|   Embedding-OpenAI-Chroma.ipynb   | Creates OpenAI Chroma vector database ~ several hours to execute, 5 GB size. Alternatively, download here: [Download Chroma database](https://www.dropbox.com/scl/fi/237x8upy01vy8v6kw7h9i/Chroma.zip?rlkey=0dga7zqksbz22pwq1sqzkj02f&dl=0) | None
|   Hybrid-Search-Contextual-Compression.ipynb   |Takes question, retrieves relevant docs with hybrid search, does contextual compression, generates answer with LLM | Chroma database & papers.json |
|   LLM-GPT3.5-Turbo.ipynb   | Uses only Chroma database and invokes LLM on question|  Chroma database & papers.json |


## Installation of our final QA-system

  ### Test the model in browser:
  1. Join the HuggingSpace Organization : [Huggingface Organization](https://huggingface.co/organizations/inltp-group20/share/sTBJmwoxoUamGbTXfJnIeqAEtsyqAggWgg)
  2. Test the model here : [PubMed Model](https://huggingface.co/spaces/inltp-group20/inltp_group20_pubmed_model) (might take around 3 minutes to generate answers)

  ### Run the app in localhost:
  1. Clone the repo : git clone https://github.com/nokitoino/_Q-A-INLPT-WS2023.git
  2. Download the chroma_openai embeddings file (5.8 GB) from here : [chroma_openai](https://www.dropbox.com/scl/fi/237x8upy01vy8v6kw7h9i/Chroma.zip?rlkey=0dga7zqksbz22pwq1sqzkj02f&dl=0)
  3. Navigate to ```Frontend-Stremlit/Hybrid_search.py```
  4. Change the path of ```'./chroma_openai'``` to the new path.
  5. Add the *OPENAI_API_KEY*
  6. Inside terminal run ```pip install streamlit```
  7. Run the app by ```streamlit run app.py```   (Give path to the app.py)



## License

This project is licensed under the [GNU General Public License (GPL) version 3](LICENSE.md) - see the [LICENSE.md](LICENSE.md) file for details.

