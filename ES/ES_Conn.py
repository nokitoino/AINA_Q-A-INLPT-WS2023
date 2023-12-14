from  elasticsearch import  Elasticsearch
es = Elasticsearch('https://localhost:9200', basic_auth=('elastic', 'jFIXIIYb73eHTwY5QWhI'), ca_certs='/Users/anureddy/Desktop/SEM03/Natural_Language_Processing_with_Transformers/Project_QA/elasticsearch-8.11.1/config/certs/http_ca.crt')
print(es.ping())