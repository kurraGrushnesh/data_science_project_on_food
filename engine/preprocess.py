import pandas as pd
from gensim.models import Word2Vec


def clean_data():
    df = pd.read_json("data/raw/recipes.json")
    df["ingredients"] = df["ingredients"].apply(lambda x: [i.lower() for i in x])
    df.to_csv("data/processed/recipes.csv", index=False)

    # Train word embeddings
    model = Word2Vec(df["ingredients"], vector_size=100, window=5, min_count=1)
    model.save("models/embeddings/food2vec.model")