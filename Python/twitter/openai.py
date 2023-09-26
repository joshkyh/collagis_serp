import os
import openai
import pandas as pd
from tqdm import tqdm
import pandas as pd


# Move get_embedding to the global scope
def get_embedding(text_to_embed):
    embedding = openai.Embedding.create(
        input=text_to_embed, model="text-embedding-ada-002"
    )["data"][0]["embedding"]

    return embedding


def get_ada_embeddings():
    # Get the reviews
    df = pd.concat([pd.read_parquet(f'data/twitter/{file}') for file in os.listdir('data/twitter/')], ignore_index=True)

    # Get non-null reviews
    df = df[df['text'].notnull()]

    # Get Quartile 1 character length
    df['text'].str.len().quantile(0.25)
    df['text'].str.len().quantile(0.50)
    df['text'].str.len().quantile(0.75)

    # Get OPENAI KEY
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # Initialize tqdm with pandas
    tqdm.pandas()

    # Apply the function and show a progress bar
    df["embedding"] = df["text"].astype(str).progress_apply(get_embedding)

    # Write the reviews to a parquet file
    df.to_parquet('data/twitter/twitter_with_embeddings.parquet', engine='pyarrow')



if __name__ == '__main__':
    get_ada_embeddings()

