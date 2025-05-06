# SPDX-License-Identifier: Apache-2.0
"""
RAG (Retrieval Augmented Generation) Implementation with LlamaIndex
================================================================

This script demonstrates a RAG system using:
- LlamaIndex: For document indexing and retrieval
- Milvus: As vector store backend
- vLLM: For embedding and text generation

Features:
1. Document Loading & Processing
2. Embedding & Storage
3. Query Processing

Requirements:
1. Install dependencies:
pip install llama-index llama-index-readers-web \
            llama-index-llms-openai-like    \
            llama-index-embeddings-openai-like \
            llama-index-vector-stores-milvus \

2. Start services:
    # Start embedding service (port 8000)
    vllm serve ssmits/Qwen2-7B-Instruct-embed-base

    # Start chat service (port 8001)
    vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001

Usage:
    python retrieval_augmented_generation_with_llamaindex.py

Notes:
    - Ensure both vLLM services are running before executing
    - Default ports: 8000 (embedding), 8001 (chat)
    - First run may take time to download models
"""

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.readers.web import SimpleWebPageReader
from llama_index.vector_stores.milvus import MilvusVectorStore


def load_documents(url: str) -> list:
    """Load and process web documents"""
    return SimpleWebPageReader(html_to_text=True).load_data([url])


def setup_models(embedding_endpoint: str,
                 chat_endpoint: str,
                 embedding_model: str,
                 chat_model: str,
                 api_key: str = "EMPTY"):
    """Configure embedding and chat models"""
    Settings.embed_model = OpenAILikeEmbedding(
        api_base=embedding_endpoint,
        api_key=api_key,
        model_name=embedding_model,
    )

    Settings.llm = OpenAILike(
        model=chat_model,
        api_key=api_key,
        api_base=chat_endpoint,
        context_window=128000,
        is_chat_model=True,
        is_function_calling_model=False,
    )

    Settings.transformations = [
        SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200,
        )
    ]


def setup_vector_store(db_path: str) -> MilvusVectorStore:
    """Initialize vector store"""
    sample_emb = Settings.embed_model.get_text_embedding("test")
    print(f"Embedding dimension: {len(sample_emb)}")
    return MilvusVectorStore(uri=db_path, dim=len(sample_emb), overwrite=True)


def create_index(documents: list, vector_store: MilvusVectorStore):
    """Create document index"""
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )


def query_document(index: VectorStoreIndex, question: str, top_k: int = 3):
    """Query document with given question"""
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    return query_engine.query(question)


def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(
        description='RAG with vLLM and LlamaIndex')

    # Add command line arguments
    parser.add_argument(
        '--url',
        default=("https://docs.vllm.ai/en/latest/getting_started/"
                 "quickstart.html"),
        help='URL of the document to process')
    parser.add_argument('--embedding-model',
                        default="ssmits/Qwen2-7B-Instruct-embed-base",
                        help='Model name for embeddings')
    parser.add_argument('--chat-model',
                        default="qwen/Qwen1.5-0.5B-Chat",
                        help='Model name for chat')
    parser.add_argument('--vllm-api-key',
                        default="EMPTY",
                        help='API key for vLLM compatible services')
    parser.add_argument('--embedding-endpoint',
                        default="http://localhost:8000/v1",
                        help='Base URL for embedding service')
    parser.add_argument('--chat-endpoint',
                        default="http://localhost:8001/v1",
                        help='Base URL for chat service')
    parser.add_argument('--db-path',
                        default="./milvus_demo.db",
                        help='Path to Milvus database')
    parser.add_argument('-i',
                        '--interactive',
                        action='store_true',
                        help='Enable interactive Q&A mode')

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Load documents
    documents = load_documents(args.url)

    # Setup models
    setup_models(args.embedding_endpoint, args.chat_endpoint,
                 args.embedding_model, args.chat_model, args.vllm_api_key)

    # Setup vector store
    vector_store = setup_vector_store(args.db_path)

    # Create index
    index = create_index(documents, vector_store)

    if args.interactive:
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            # Get user question
            question = input("\nEnter your question: ")

            # Check for exit command
            if question.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode...")
                break

            # Get and print response
            print("\n" + "-" * 50)
            print("Response:\n")
            response = query_document(index, question)
            print(response)
            print("-" * 50)
    else:
        # Single query mode
        question = "How to install vLLM?"
        response = query_document(index, question)
        print("-" * 50)
        print("Response:\n")
        print(response)
        print("-" * 50)


if __name__ == "__main__":
    main()
