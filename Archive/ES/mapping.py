indexMapping={
      "properties": {
        "abstract": {
          "properties": {
            "full_text": {
              "type": "text"
            }
          }
        },
        "abstract_encoded": {
          "type": "dense_vector",
          "dims": 768,
          "index": True,
          "similarity": "l2_norm"
        },
        "affiliations": {
          "type": "keyword",
          "ignore_above": 256            
            },
        "authors": {
          "type": "keyword",
          "ignore_above": 256         
            },
        "identifiers": {
          "properties": {
            "doi": {
              "type": "text",
              "fields": {
                "keyword": {
                  "type": "keyword",
                  "ignore_above": 256
                }
              }
            },
            "medline": {
                "type": "keyword",
                "ignore_above": 256
                },
            "mid": {
                "type": "keyword",
                "ignore_above": 256
            },
            "pii": {
              "type": "text",
              "fields": {
                "keyword": {
                  "type": "keyword",
                  "ignore_above": 256
                }
              }
            },
            "pmc": {
                  "type": "keyword",
                  "ignore_above": 256
                },
            "pubmed": {
              "type": "long"
                },
            "sici": {
              "type": "text",
              "fields": {
                "keyword": {
                  "type": "keyword",
                  "ignore_above": 256
                }
              }
            }
          }
        },
        "journal": {
          "type": "text"
        },
        "keywords": {
          "type": "text",
          "fields": {
            "keyword": {
              "type": "keyword",
              "ignore_above": 256
            }
          }
        },
        "language": {
          "type": "text"
        },
        "month": {
          "type": "text"
        },
        "references": {
          "type": "text"
        },
        "title": {
          "properties": {
            "full_text": {
              "type": "text",
              "fields": {
                "keyword": {
                  "type": "keyword",
                  "ignore_above": 256
                }
              }
            }
          }
        },
        "year": {
          "type": "date"
        }
      }
    }