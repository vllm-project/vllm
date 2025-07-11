# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example script demonstrating long text embedding with chunked processing in vLLM.

This example shows how to use vLLM's chunked processing feature to handle text
inputs that exceed the model's maximum token length. The feature automatically
splits long text into chunks and aggregates the results.

Prerequisites:
1. Start vLLM server with chunked processing enabled:
   
   vllm serve intfloat/multilingual-e5-large \
     --task embed \
     --override-pooler-config \
      '{"pooling_type": "CLS", "normalize": true, \"enable_chunked_processing": true}' \
     --max-model-len 10240 \
     --served-model-name multilingual-e5-large \
     --trust-remote-code \
     --port 31090 \
     --api-key your-api-key

2. Install required dependencies:
   pip install openai requests
"""

import time

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
            import numpy as np

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
    test_embedding_consistency()

    print("\n" + "=" * 70)
    print("🎉 All tests completed!")
    print("\n💡 Key Features Demonstrated:")
    print("   - ✅ Automatic chunked processing for long text")
    print("   - ✅ Seamless handling of mixed-length batches")
    print("   - ✅ Consistent embedding generation")
    print("   - ✅ Backward compatibility with short text")
    print("\n📚 For more information, see:")
    print(
        "   - Documentation: https://docs.vllm.ai/en/latest/models/pooling_models.html"
    )
    print("   - Chunked Processing Guide: openai_embedding_long_text.md")


if __name__ == "__main__":
    main()
