import pandas as pd

df_sgl = pd.read_json("sgl-results.json", lines=True)
df_sgl.to_csv("sgl-results.csv")

df_vllm = pd.read_json("vllm-results.json", lines=True)
df_vllm.to_csv("vllm-results.csv")
