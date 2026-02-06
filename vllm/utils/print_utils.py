



def print_embeddings(embeds: list[float]):
    embeds_trimmed = (str(embeds[:4])[:-1] + ", ...]") if len(embeds) > 4 else embeds
    print(f"Embeddings: {embeds_trimmed} (size={len(embeds)})")