from fastembed import TextEmbedding
import os
import json
import logging
from groq import Groq, APIConnectionError, RateLimitError, APIStatusError
from pinecone.exceptions import PineconeException

from src.upsert import get_embeddings, pinecone_client, start_pc_index
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def query_index(index, namespace, query_vector):
    try:
        results = index.query(
            namespace=namespace,
            vector=query_vector,
            top_k=2,
            include_metadata=True
        )
    except PineconeException as e:
        logger.error(f"Pinecone API error: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return []
    for match in results['matches']:
        logger.info(f"Score: {match['score']:.4f} | Text: {match['metadata']['text']}")
    matches = ["\n".join(match['metadata']['text'] for match in results['matches'])]
    return matches

def llm_call(model_name, context, user_input):
    client = Groq(
        api_key=os.environ.get('LLM_API_KEY')
    )
    with open(os.path.join(os.path.dirname(__file__), 'prompts', 'system.prompt.md'), 'r') as system_prompt_file:
        system_prompt = system_prompt_file.read()
    
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Context: " + "\n".join(context) + "\n\nUser input: " + user_input}
            ],
            temperature=1,
            max_tokens=1024
        )
        response = completion.choices[0].message.content or ""
    except APIConnectionError as e:
        logger.error(f"The server could not be reached: {e.__cause__}")
        return "Error: Unable to reach the server at the moment."
    except RateLimitError as e:
        logger.error(f"A 429 status code was received; back off and retry.")
        return "Error: Rate limit exceeded. Please try again later."
    except APIStatusError as e:
        logger.error(f"Another non-200-range status code was received: {e.status_code}")
        logger.error(e.response)
        return f"Error: Unexpected error occurred. Please try again later."
    return response

def lambda_handler(event, context):
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as config_file:
        config = json.load(config_file)

    embedding_model_name = config['embeddings']['model']
    embedding_model = TextEmbedding(model_name=embedding_model_name)
    query_vector = get_embeddings(text=event["text"], model=embedding_model)

    index_name = config['vector_store']['index_name']
    namespaces = config['vector_store']['namespaces']
    pc = pinecone_client()
    index = start_pc_index(pc=pc, index_name=index_name)
    matches = query_index(index=index, namespace=namespaces[0], query_vector=query_vector)
    llm_model_name = config['llm']['model_name']
    llm_response = llm_call(model_name=llm_model_name, context=matches, 
                            user_input=event["text"])
    return {
        'statusCode': 200,
        'body': json.dumps(llm_response)
    }
