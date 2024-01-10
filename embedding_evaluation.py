import json
import random
import openai

# Set your OpenAI GPT-3 API key
openai.api_key = 'KEY'

def generate_questions(title, context):
    prompt = f"Form two question in your own words over the {context}. Answer in json format {{""question1"": '', ""question2"": ''}}, don't say anything else."
    # Careful, when you add {title} in the prompt. ChatGPT tends to make the question title-specific

    # You can adjust temperature and max tokens as per your preferences
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the chat model
        messages=[
            {"role": "system", "content": "Act as a medical expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=100,
        n=1  # Generate two questions
    )
    print(response['choices'][0]['message']['content'])
    answer = json.loads(response['choices'][0]['message']['content'])
    print(answer['question1'])
    return (title,answer['question1']),(title,answer['question2'])

def get_random_documents(json_data, num_documents=50):
    documents = json.loads(json_data)
    selected_documents = random.sample(documents, min(num_documents, len(documents)))

    result = []
    for document in selected_documents:
        title = document.get("title", {}).get("full_text", "")
        context = document.get("abstract", {}).get("full_text", "")
        result.append((title, context))

    return result

with open('papers.json', 'r') as file:
    json_data = file.read()

rand_docs = get_random_documents(json_data, num_documents=1)

questions_result = [] # Array of (title, question)
for title, context in rand_docs:
    generated_questions = generate_questions(title, context)
    questions_result.extend(generated_questions) # Extend vs append: We wan't the result flat.

print(questions_result)
