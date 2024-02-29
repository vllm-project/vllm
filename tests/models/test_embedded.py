from vllm.model_executor.models.embedded_model import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

# embeddings_1 = model.encode(sentences_1, 
#                             batch_size=12, 
#                             max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
#                             )
# embeddings_2 = model.encode(sentences_2)
# similarity = embeddings_1 @ embeddings_2.T
# print(similarity)
