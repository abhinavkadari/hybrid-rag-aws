# Hybrid RAG System (Serverless AWS)
Most RAG systems suffer from high inference costs and hallucinations. This project implements a Hybrid RAG approach using BGE-Small embeddings and Groq-accelerated Llama 3 to provide low-latency, grounded answers with zero server overhead.

## Engineering Decisions
- LLM Model:
  - Llama 3 hosted on Groq API: Fastest inference, utilizing an LPU
- Embedding Model: 
  - BGE-Small: MTEB Score, Model Size
  - Fastembed(Qdrant): To avoid saving large dependencies like torch

## Setup and Usage
- Prerequisites:
  - Pinecone, Groq, Unstructured APIs
- Create and push Docker images
```
containarize.cmd <aws account_id> upsert
containarize.cmd <aws account_id> generate
```
- AWS Lambda: 
  - Create Upsert, Generate Lambda functions
  - Use Container Image -> Select Container Image URI (hybrid-rag/upsert, hybrid-rag/generate)
  - Add Triger -> Source: S3 -> All object create events -> Add Bucket -> Prefix/Suffix of files to be uploaded
  - Add Permissions, Environment Variables
- Run Streamlit App
```
streamlit run .\src\app.py
```