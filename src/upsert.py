import boto3
from botocore.exceptions import ClientError
import io
import os
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException
import json
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def s3_file_fetch(bucket_name, file_key):
    logger.info("Fetching file from S3")
    s3 = boto3.client('s3')
    try:
        s3_object = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_content = s3_object['Body'].read()
        file_stream = io.BytesIO(file_content)
        return file_stream
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDenied':
            logger.error(f"Access denied for bucket: {bucket_name}")
        elif error_code == 'InvalidLocationConstraint':
            logger.error(f"File not found: {file_key} in bucket: {bucket_name}")
        raise

def load_unstructured_data(file_stream):
    logger.info("Initializing UnstructuredLoader")
    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    if not api_key:
        logger.error("UNSTRUCTURED_API_KEY environment variable is not set")
        raise ValueError("UNSTRUCTURED_API_KEY environment variable is required")
    try:
        loader = UnstructuredLoader(
            file=file_stream,
            partition_via_api=True,
            api_key=api_key,
            strategy="fast"
        )
    except Exception as e:
        logger.error(f"Error occurred while initializing UnstructuredLoader: {e}")
        loader = PyPDFLoader(file=file_stream)
    # Load the text content
    data = loader.load()
    return data

def create_chunks(data):
    logger.info("Creating chunks")
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        logger.info(f"Total chunks created: {len(chunks)}")
        logger.info(chunks[0].page_content)
        return chunks
    except TypeError as e:
        logger.error(f"Configuration error: Check your parameter types: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during chunking: {e}")
        raise

def get_embeddings(text, model):
    instruction = "Represent this sentence for searching relevant passages: "
    vector = next(model.embed([instruction + text])).tolist()
    return vector

def embeddings_from_chunks(chunks, embedding_model):
    logger.info("Creating embeddings from chunks")
    vectors = []
    for i, chunk in enumerate(chunks):
        vector = {"id": "vec{}".format(i), "values": get_embeddings(text=chunk.page_content, model=embedding_model), 
                  "metadata": {"text": chunk.page_content}}
        vectors.append(vector)
    return vectors

def pinecone_client():
    logger.info("Connecting to Pinecone")
    api_key = os.getenv('API_KEY')
    if not api_key:
        logger.error("API_KEY environment variable is not set")
        raise ValueError("API_KEY environment variable is required")
    pc = Pinecone(api_key=api_key)
    return pc

def create_pc_index(pc, index_name):
    logger.info("Creating index")
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384, 
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

def start_pc_index(pc, index_name):
    logger.info("Starting index")
    index = pc.Index(index_name)
    return index

def delete_data_from_index(index, namespace):
    logger.info("Deleting data from index")
    index.delete(delete_all=True, namespace=namespace)

def upsert_data(index, vectors, namespace):
    logger.info("Upserting data")
    try:
        index.upsert(
            vectors=vectors,
            namespace=namespace
        )
    except PineconeException as e:
        logger.error(f"Pinecone API error during upsert: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upsert: {e}")
        raise

def lambda_handler(event, context):
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as config_file:
        config = json.load(config_file)

    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_key = event['Records'][0]['s3']['object']['key']
    file_stream = s3_file_fetch(bucket_name=bucket_name, file_key=file_key)
    data = load_unstructured_data(file_stream=file_stream)
    chunks = create_chunks(data=data)

    embedding_model_name = config['embeddings']['model']
    embedding_model = TextEmbedding(model_name=embedding_model_name)
    vectors = embeddings_from_chunks(chunks=chunks, embedding_model=embedding_model)

    index_name = config['vector_store']['index_name']
    namespaces = config['vector_store']['namespaces']
    pc = pinecone_client()
    create_pc_index(pc=pc, index_name=index_name)
    index = start_pc_index(pc=pc, index_name=index_name)
    # delete_data_from_index(index=index, namespace=namespaces[0])
    upsert_data(index=index, vectors=vectors, namespace=namespaces[0])
    
    return {
        'statusCode': 200,
        'body': json.dumps(str(event))
    }
