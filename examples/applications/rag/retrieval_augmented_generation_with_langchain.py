# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Retrieval Augmented Generation (RAG) Implementation with Langchain
==================================================================

This script demonstrates a RAG implementation using LangChain, Milvus
and vLLM. RAG enhances LLM responses by retrieving relevant context
from a document collection.

Features:
- Web content loading and chunking
- Vector storage with Milvus
- Embedding generation with vLLM
- Question answering with context

Prerequisites:
1. Install dependencies:
    pip install -U vllm \
                 langchain_milvus langchain_openai \
                 langchain_community beautifulsoup4 \
                 langchain-text-splitters

2. Start services:
    # Start embedding service (port 8000)
    vllm serve ssmits/Qwen2-7B-Instruct-embed-base

    # Start chat service (port 8001)
    vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001

Usage:
    python retrieval_augmented_generation_with_langchain.py

Notes:
    - Ensure both vLLM services are running before executing
    - Default ports: 8000 (embedding), 8001 (chat)
    - First run may take time to download models
"""

import argparse
from argparse import Namespace
from typing import Any

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_documents(config: dict[str, Any]):
    """
    Load and split documents from web URL
    """
    try:
        loader = WebBaseLoader(web_paths=(config["url"],))
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )
        return text_splitter.split_documents(docs)
    except Exception as e:
        print(f"Error loading document from {config['url']}: {str(e)}")
        raise


def init_vectorstore(config: dict[str, Any], documents: list[Document]):
    """
    Initialize vector store with documents
    """
    return Milvus.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(
            model=config["embedding_model"],
            openai_api_key=config["vllm_api_key"],
            openai_api_base=config["vllm_embedding_endpoint"],
        ),
        connection_args={"uri": config["uri"]},
        drop_old=True,
    )


def init_llm(config: dict[str, Any]):
    """
    Initialize llm
    """
    return ChatOpenAI(
        model=config["chat_model"],
        openai_api_key=config["vllm_api_key"],
        openai_api_base=config["vllm_chat_endpoint"],
    )


def get_qa_prompt():
    """
    Get question answering prompt template
    """
    template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
    return PromptTemplate.from_template(template)


def format_docs(docs: list[Document]):
    """
    Format documents for prompt
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_qa_chain(retriever: Any, llm: ChatOpenAI, prompt: PromptTemplate):
    """
    Set up question answering chain
    """
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def get_parser() -> argparse.ArgumentParser:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="RAG with vLLM and langchain")

    # Add command line arguments
    parser.add_argument(
        "--vllm-api-key", default="EMPTY", help="API key for vLLM compatible services"
    )
    parser.add_argument(
        "--vllm-embedding-endpoint",
        default="http://localhost:8000/v1",
        help="Base URL for embedding service",
    )
    parser.add_argument(
        "--vllm-chat-endpoint",
        default="http://localhost:8001/v1",
        help="Base URL for chat service",
    )
    parser.add_argument("--uri", default="./milvus.db", help="URI for Milvus database")
    parser.add_argument(
        "--url",
        default=("https://docs.vllm.ai/en/latest/getting_started/quickstart.html"),
        help="URL of the document to process",
    )
    parser.add_argument(
        "--embedding-model",
        default="ssmits/Qwen2-7B-Instruct-embed-base",
        help="Model name for embeddings",
    )
    parser.add_argument(
        "--chat-model", default="qwen/Qwen1.5-0.5B-Chat", help="Model name for chat"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Enable interactive Q&A mode"
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=3, help="Number of top results to retrieve"
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for document splitting",
    )
    parser.add_argument(
        "-o",
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for document splitting",
    )

    return parser


def init_config(args: Namespace):
    """
    Initialize configuration settings from command line arguments
    """

    return {
        "vllm_api_key": args.vllm_api_key,
        "vllm_embedding_endpoint": args.vllm_embedding_endpoint,
        "vllm_chat_endpoint": args.vllm_chat_endpoint,
        "uri": args.uri,
        "embedding_model": args.embedding_model,
        "chat_model": args.chat_model,
        "url": args.url,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "top_k": args.top_k,
    }


def main():
    # Parse command line arguments
    args = get_parser().parse_args()

    # Initialize configuration
    config = init_config(args)

    # Load and split documents
    documents = load_and_split_documents(config)

    # Initialize vector store and retriever
    vectorstore = init_vectorstore(config, documents)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config["top_k"]})

    # Initialize llm and prompt
    llm = init_llm(config)
    prompt = get_qa_prompt()

    # Set up QA chain
    qa_chain = create_qa_chain(retriever, llm, prompt)

    # Interactive mode
    if args.interactive:
        print("\nWelcome to Interactive Q&A System!")
        print("Enter 'q' or 'quit' to exit.")

        while True:
            question = input("\nPlease enter your question: ")
            if question.lower() in ["q", "quit"]:
                print("\nThank you for using! Goodbye!")
                break

            output = qa_chain.invoke(question)
            print(output)
    else:
        # Default single question mode
        question = "How to install vLLM?"
        output = qa_chain.invoke(question)
        print("-" * 50)
        print(output)
        print("-" * 50)


if __name__ == "__main__":
    main()
