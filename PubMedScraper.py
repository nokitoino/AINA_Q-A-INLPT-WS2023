import pandas as pd
from Bio import Entrez
import json
# Set your email address for the Entrez API
Entrez.email = "YOUR EMAIL HERE"


def fetch_details(id_list):
    ids = ','.join(id_list)
    handle = Entrez.efetch(db='pubmed', id=ids, retmode='xml')
    papers = Entrez.read(handle)
    return papers


# Your search query parameters
search_query = "intelligence"
start_date = "2014/01/01"
end_date = "2014/12/31"

# Use esearch to get the list of IDs matching your criteria
search_handle = Entrez.esearch(db='pubmed', term=search_query, mindate=start_date, maxdate=end_date, retmax=100000)
search_results = Entrez.read(search_handle)
search_handle.close()

# Extract the list of IDs
studies_id_list = search_results['IdList']

# Initialize lists to store data
data_list = []

chunk_size = 10000
# Fetch details in chunks
for chunk_i in range(0, len(studies_id_list), chunk_size):
    chunk = studies_id_list[chunk_i:chunk_i + chunk_size]
    papers = fetch_details(chunk)
    for i, paper in enumerate(papers['PubmedArticle']):
        data = {
            "title": {
                "full_text": "",
                "tokens": []
            },
            "authors": [],
            "affiliations": [],
            "identifiers": {},
            "journal": "",
            "language": "",
            "abstract": {
                "full_text": "",
                "tokens": []
            },
            "year": "",
            "month": "",
            "keywords": [],
            "references": []
        }
        try:
            data["title"]["full_text"] = paper['MedlineCitation']['Article']['ArticleTitle'].lower()
            data["title"]["tokens"] = data["title"]["full_text"].split() #Use specil Tokenizer her, for removing stopwords ect.
        except Exception as e:
            print(f"Error processing study {i}: {e}")
        try:
            data["abstract"]["full_text"] = paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0].lower()
            data["abstract"]["tokens"] = data["abstract"]["full_text"].split()
        except:
            data["abstract"]["full_text"] = ''
        data["journal"] = paper['MedlineCitation']['Article']['Journal']['Title'].lower()
        data["language"] = paper['MedlineCitation']['Article']['Language'][0].lower()
        try:
                data["year"] = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year'].lower()
        except:
            data["year"] = ""
        try:
            data["month"] = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Month'].lower()
        except:
            data["month"] = ""

        # Extracting Keywords
        keywords = []
        if 'KeywordList' in paper['MedlineCitation']:
            for keyword_list in paper['MedlineCitation']['KeywordList']:
                keywords.append(keyword_list)
        data["keywords"] = keywords

        # Extracting References
        references = []
        if 'PubmedData' in paper and 'ReferenceList' in paper['PubmedData']:
            for reference in paper['PubmedData']['ReferenceList']:
                for citation in reference['Reference']:
                    references.append(citation['Citation'])  # or by ArticleIdList we can access the DOI directly
        data["references"] = references
        data_list.append(data)

# Save the data to a JSON file
with open('papers.json', 'w') as json_file:
    json.dump(data_list, json_file, indent=2)