import pandas as pd
from Bio import Entrez
import json
from datetime import datetime, timedelta

# Set your email address for the Entrez API
Entrez.email = "gustafimeri@gmail.com"


def fetch_details(id_list):
    ids = ','.join(id_list)
    handle = Entrez.efetch(db='pubmed', id=ids, retmode='xml')
    papers = Entrez.read(handle)
    return papers

def generate_date_intervals(start_date, end_date, days):
    date_format = "%Y/%m/%d"

    start_datetime = datetime.strptime(start_date, date_format)
    end_datetime = datetime.strptime(end_date, date_format)

    current_datetime = start_datetime
    date_intervals = []

    while current_datetime < end_datetime:
        date_intervals.append(current_datetime.strftime(date_format))
        current_datetime += timedelta(days=days)

    # Add the end date to ensure the last interval covers the remaining days
    date_intervals.append(end_datetime.strftime(date_format))

    return date_intervals

# Your search query parameters
search_query = "intelligence"
start_date = "2014/01/01"
end_date = "2014/3/24"
days_interval = 200

# Generate date intervals
intervals = generate_date_intervals(start_date, end_date, days_interval)

# Initialize lists to store data
data_list = []

chunk_size = 10000

# Loop through date intervals
for interval_start, interval_end in zip(intervals[:-1], intervals[1:]):
    # Use esearch to get the list of IDs matching your criteria for the current interval
    search_handle = Entrez.esearch(db='pubmed', term=search_query, mindate=interval_start, maxdate=interval_end, retmax=60000)
    search_results = Entrez.read(search_handle)
    search_handle.close()

    # Extract the list of IDs
    studies_id_list = search_results['IdList']
    print(interval_start,interval_end,len(studies_id_list))
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
                data["title"]["tokens"] = data["title"]["full_text"].split()  # Use specil Tokenizer here, for removing stopwords ect.
                for author in paper['MedlineCitation']['Article']['AuthorList']:
                    author_name = f"{author.get('LastName', '')}, {author.get('ForeName', '')}"
                    data["authors"].append(author_name)
            except:
                ...
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

            # Extracting Affiliations
            affiliations = []
            if 'AuthorList' in paper['MedlineCitation']['Article']:
                author_list = paper['MedlineCitation']['Article']['AuthorList']
                for author_info in author_list:
                    if 'AffiliationInfo' in author_info:
                        for affiliation_info in author_info['AffiliationInfo']:
                            affiliation = affiliation_info.get('Affiliation', '')
                            affiliations.append(affiliation)
            else:
                ...
                #print(f"No affiliation information for paper {i}")

            data["affiliations"] = affiliations

            # Identifiers (e.g., DOI, PMID)
            identifiers = {}
            if 'PubmedData' in paper and 'ArticleIdList' in paper['PubmedData']:
                for identifier in paper['PubmedData']['ArticleIdList']:
                    data["identifiers"][identifier.attributes['IdType']] = str(identifier)

            data_list.append(data)

# Save the data to a JSON file
with open('papers.json', 'w') as json_file:
    json.dump(data_list, json_file, indent=2)
