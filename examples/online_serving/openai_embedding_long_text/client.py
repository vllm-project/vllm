# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example script demonstrating long text embedding with chunked processing in vLLM.

This example shows how to use vLLM's chunked processing feature to handle text
inputs that exceed the model's maximum token length. The feature automatically
splits long text into chunks and handles different pooling types optimally.

Prerequisites:
1. Start vLLM server with chunked processing enabled:
   
   # MEAN pooling (processes all chunks, recommended for complete coverage)
   vllm serve intfloat/multilingual-e5-large \
     --pooler-config \
      '{"pooling_type": "MEAN", "normalize": true, ' \
      '"enable_chunked_processing": true, "max_embed_len": 3072000}' \
     --served-model-name multilingual-e5-large \
     --trust-remote-code \
     --port 31090 \
     --api-key your-api-key

   # OR CLS pooling (native CLS within chunks, MEAN aggregation across chunks)
   vllm serve BAAI/bge-large-en-v1.5 \
     --pooler-config \
      '{"pooling_type": "CLS", "normalize": true, ' \
      '"enable_chunked_processing": true, "max_embed_len": 1048576}' \
     --served-model-name bge-large-en-v1.5 \
     --trust-remote-code \
     --port 31090 \
     --api-key your-api-key

2. Install required dependencies:
   pip install openai requests
"""

import time

import numpy as np
from openai import OpenAI

# Configuration
API_KEY = "your-api-key"  # Replace with your actual API key
BASE_URL = "http://localhost:31090/v1"
MODEL_NAME = "multilingual-e5-large"


def generate_long_text(base_text: str, repeat_count: int) -> str:
    """Generate long text by repeating base text."""
    return base_text * repeat_count


def test_embedding_with_different_lengths():
    """Test embedding generation with different text lengths."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Test cases with different text lengths
    test_cases = [
        {
            "name": "Short Text",
            "text": "Hello, this is a short text for embedding.",
            "expected_chunks": 1,
        },
        {
            "name": "Medium Text",
            "text": generate_long_text(
                "This is a medium-length text that should fit within the "
                "model's context window. " * 20,
                2,
            ),
            "expected_chunks": 1,
        },
        {
            "name": "Long Text (2 chunks)",
            "text": generate_long_text(
                "This is a very long text that will exceed the model's "
                "maximum context length and trigger chunked processing. " * 50,
                5,
            ),
            "expected_chunks": 2,
        },
        {
            "name": "Very Long Text (3+ chunks)",
            "text": generate_long_text(
                "This text is extremely long and will definitely "
                "require multiple chunks for processing. " * 100,
                10,
            ),
            "expected_chunks": 3,
        },
    ]

    print("🧪 Testing vLLM Long Text Embedding with Chunked Processing")
    print("=" * 70)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print(f"Text length: {len(test_case['text'])} characters")

        try:
            start_time = time.time()

            response = client.embeddings.create(
                input=test_case["text"], model=MODEL_NAME, encoding_format="float"
            )

            end_time = time.time()
            processing_time = end_time - start_time

            # Extract embedding data
            embedding = response.data[0].embedding
            embedding_dim = len(embedding)

            print("✅ Success!")
            print(f"   - Embedding dimension: {embedding_dim}")
            print(f"   - Processing time: {processing_time:.2f}s")
            print(f"   - Expected chunks: ~{test_case['expected_chunks']}")
            print(f"   - First 5 values: {embedding[:5]}")

        except Exception as e:
            print(f"❌ Failed: {str(e)}")


def test_batch_embedding():
    """Test batch embedding with mixed-length inputs."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print("\n🔄 Testing Batch Embedding with Mixed Lengths")
    print("=" * 50)

    # Mix of short and long texts
    batch_inputs = [
        "Short text 1",
        generate_long_text("Medium length text that fits in one chunk. " * 20, 1),
        "Another short text",
        generate_long_text("Long text requiring chunked processing. " * 100, 5),
    ]

    try:
        start_time = time.time()

        response = client.embeddings.create(
            input=batch_inputs, model=MODEL_NAME, encoding_format="float"
        )

        end_time = time.time()
        processing_time = end_time - start_time

        print("✅ Batch processing successful!")
        print(f"   - Number of inputs: {len(batch_inputs)}")
        print(f"   - Number of embeddings: {len(response.data)}")
        print(f"   - Total processing time: {processing_time:.2f}s")
        print(
            f"   - Average time per input: {processing_time / len(batch_inputs):.2f}s"
        )

        for i, data in enumerate(response.data):
            input_length = len(batch_inputs[i])
            embedding_dim = len(data.embedding)
            print(
                f"   - Input {i + 1}: {input_length} chars → {embedding_dim}D embedding"
            )

    except Exception as e:
        print(f"❌ Batch processing failed: {str(e)}")


def test_multiple_long_texts_batch():
    """Test batch processing with multiple long texts to verify chunk ID uniqueness."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print("\n🔧 Testing Multiple Long Texts in Batch (Chunk ID Fix Verification)")
    print("=" * 70)

    # Create multiple distinct long texts that will all require chunking
    # Note: All pooling types now use MEAN aggregation across chunks:
    # - Native pooling (MEAN/CLS/LAST) is used within each chunk
    # - MEAN aggregation combines results across all chunks
    # - Full semantic coverage for all pooling types
    long_texts = [
        generate_long_text(
            "First long document about artificial intelligence and machine learning. "
            * 80,
            6,
        ),
        generate_long_text(
            "Second long document about natural language processing and transformers. "
            * 80,
            6,
        ),
        generate_long_text(
            "Third long document about computer vision and neural networks. " * 80, 6
        ),
    ]

    # Add some short texts to mix things up
    batch_inputs = [
        "Short text before long texts",
        long_texts[0],
        "Short text between long texts",
        long_texts[1],
        long_texts[2],
        "Short text after long texts",
    ]

    print("📊 Batch composition:")
    for i, text in enumerate(batch_inputs):
        length = len(text)
        text_type = "Long (will be chunked)" if length > 5000 else "Short"
        print(f"   - Input {i + 1}: {length} chars ({text_type})")

    try:
        start_time = time.time()

        response = client.embeddings.create(
            input=batch_inputs, model=MODEL_NAME, encoding_format="float"
        )

        end_time = time.time()
        processing_time = end_time - start_time

        print("\n✅ Multiple long texts batch processing successful!")
        print(f"   - Number of inputs: {len(batch_inputs)}")
        print(f"   - Number of embeddings returned: {len(response.data)}")
        print(f"   - Total processing time: {processing_time:.2f}s")

        # Verify each embedding is different (no incorrect aggregation)
        embeddings = [data.embedding for data in response.data]

        if len(embeddings) >= 3:
            import numpy as np

            # Compare embeddings of the long texts (indices 1, 3, 4)
            long_embeddings = [
                np.array(embeddings[1]),  # First long text
                np.array(embeddings[3]),  # Second long text
                np.array(embeddings[4]),  # Third long text
            ]

            print("\n🔍 Verifying embedding uniqueness:")
            for i in range(len(long_embeddings)):
                for j in range(i + 1, len(long_embeddings)):
                    cosine_sim = np.dot(long_embeddings[i], long_embeddings[j]) / (
                        np.linalg.norm(long_embeddings[i])
                        * np.linalg.norm(long_embeddings[j])
                    )
                    print(
                        f"   - Similarity between long text {i + 1} and {j + 1}: "
                        f"{cosine_sim:.4f}"
                    )

                    if (
                        cosine_sim < 0.9
                    ):  # Different content should have lower similarity
                        print("     ✅ Good: Embeddings are appropriately different")
                    else:
                        print(
                            "     ⚠️ High similarity - may indicate chunk "
                            "aggregation issue"
                        )

        print("\n📋 Per-input results:")
        for i, data in enumerate(response.data):
            input_length = len(batch_inputs[i])
            embedding_dim = len(data.embedding)
            embedding_norm = np.linalg.norm(data.embedding)
            print(
                f"   - Input {i + 1}: {input_length} chars → {embedding_dim}D "
                f"embedding (norm: {embedding_norm:.4f})"
            )

        print(
            "\n✅ This test verifies the fix for chunk ID collisions in "
            "batch processing"
        )
        print("   - Before fix: Multiple long texts would have conflicting chunk IDs")
        print("   - After fix: Each prompt's chunks have unique IDs with prompt index")

    except Exception as e:
        print(f"❌ Multiple long texts batch test failed: {str(e)}")
        print("   This might indicate the chunk ID collision bug is present!")


def test_embedding_consistency():
    """Test that chunked processing produces consistent results."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print("\n🔍 Testing Embedding Consistency")
    print("=" * 40)

    # Use the same long text multiple times
    long_text = generate_long_text(
        "Consistency test text for chunked processing validation. " * 50, 3
    )

    embeddings = []

    try:
        for i in range(3):
            response = client.embeddings.create(
                input=long_text, model=MODEL_NAME, encoding_format="float"
            )
            embeddings.append(response.data[0].embedding)
            print(f"   - Generated embedding {i + 1}")

        # Check consistency (embeddings should be identical)
        if len(embeddings) >= 2:
            # Calculate similarity between first two embeddings

            emb1 = np.array(embeddings[0])
            emb2 = np.array(embeddings[1])

            # Cosine similarity
            cosine_sim = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )

            print("✅ Consistency test completed!")
            print(f"   - Cosine similarity between runs: {cosine_sim:.6f}")
            print("   - Expected: ~1.0 (identical embeddings)")

            if cosine_sim > 0.999:
                print("   - ✅ High consistency achieved!")
            else:
                print("   - ⚠️ Consistency may vary due to numerical precision")

    except Exception as e:
        print(f"❌ Consistency test failed: {str(e)}")


def main():
    """Main function to run all tests."""
    print("🚀 vLLM Long Text Embedding Client")
    print(f"📡 Connecting to: {BASE_URL}")
    print(f"🤖 Model: {MODEL_NAME}")
    masked_key = "*" * (len(API_KEY) - 4) + API_KEY[-4:] if len(API_KEY) > 4 else "****"
    print(f"🔑 API Key: {masked_key}")

    # Run all test cases
    test_embedding_with_different_lengths()
    test_batch_embedding()
    test_multiple_long_texts_batch()
    test_embedding_consistency()

    print("\n" + "=" * 70)
    print("🎉 All tests completed!")
    print("\n💡 Key Features Demonstrated:")
    print("   - ✅ Automatic chunked processing for long text")
    print("   - ✅ Seamless handling of mixed-length batches")
    print("   - ✅ Multiple long texts in single batch (chunk ID fix)")
    print("   - ✅ Unified chunked processing:")
    print("     • Native pooling used within each chunk")
    print("     • MEAN aggregation across all chunks")
    print("     • Complete semantic coverage for all pooling types")
    print("   - ✅ Consistent embedding generation")
    print("   - ✅ Backward compatibility with short text")
    print("\n📚 For more information, see:")
    print(
        "   - Documentation: https://docs.vllm.ai/en/latest/models/pooling_models.html"
    )
    print("   - Chunked Processing Guide: openai_embedding_long_text.md")


if __name__ == "__main__":
    main()
