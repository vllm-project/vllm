# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Index contextual chunks and retrieve source excerpts with multilingual E5.

Start the server with:
    vllm serve intfloat/multilingual-e5-small --runner pooling \
        --pooler-config.task token_embed --max-model-len 512

Then run this file to search an in-memory index. Use --index-path to save it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import requests


def encode(session, args, texts, chunk_size):
    response = session.post(
        f"{args.url.rstrip('/')}/pooling",
        json={
            "model": args.model,
            "input": texts,
            "task": "token_embed",
            "encoding_format": "float",
            "use_activation": True,
            "late_chunking_params": {"chunk_size": chunk_size},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["data"]


def main(args):
    # Ranges refer to these exact submitted strings, including the E5 prefix.
    documents = {
        "berlin": "passage: Berlin is the capital of Germany. The city has many "
        "museums. They display art and historical collections from around the world.",
        "kyoto": "passage: Kyoto is a city in Japan. Its historic temples attract "
        "visitors throughout the year.",
        "storage": "passage: Database backups protect data from accidental loss. "
        "They should be tested regularly to ensure that recovery works.",
    }
    chunks = []
    with requests.Session() as session:
        outputs = encode(session, args, list(documents.values()), args.chunk_size)
        for doc_id, output in zip(documents, outputs):
            for index, (vector, chunk) in enumerate(
                zip(output["data"], output["late_chunking"]["chunks"])
            ):
                if chunk["char_range"] is None:
                    continue  # No source text to retrieve for special-only chunks.
                chunks.append(
                    {
                        "document_id": doc_id,
                        "chunk_index": index,
                        "vector": vector,
                        **chunk,
                    }
                )

        # E5 uses mean pooling. One chunk covering the query gives its sequence
        # embedding, with the same model, head and normalization as the documents.
        # The server rejects queries longer than its 512-token context window.
        query = encode(session, args, [f"query: {args.query}"], chunk_size=512)[0]
        (query_embedding,) = query["data"]
        query_vector = np.asarray(query_embedding, dtype=np.float32)

    if args.index_path is not None:
        args.index_path.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "chunk_size": args.chunk_size,
                    "documents": documents,
                    "chunks": chunks,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Saved {len(chunks)} chunks to {args.index_path}")
    vectors = np.asarray([chunk["vector"] for chunk in chunks], dtype=np.float32)
    scores = vectors @ query_vector  # Cosine similarity for normalized vectors.
    for index in np.argsort(-scores)[: args.top_k]:
        chunk = chunks[index]
        start, end = chunk["char_range"]
        excerpt = documents[chunk["document_id"]][start:end]
        print(
            f"{scores[index]:.4f} {chunk['document_id']} "
            f"chunk {chunk['chunk_index']}: {excerpt!r}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--query", default="Which city has museums with art collections?"
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--index-path", type=Path, help="Optionally save the index as JSON."
    )
    main(parser.parse_args())
