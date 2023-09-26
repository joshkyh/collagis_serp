import os
from embedchain import App
import pandas as pd
from tqdm import tqdm

# Create a bot instance
bot = App()

# Read parquet
reviews = pd.read_parquet(os.path.join("data","productreviews", "dodo_01-09.parquet"))

# Embed online resources
# bot add each row of snippet
for snippet in tqdm(reviews.snippet):
    bot.add(snippet)



# Query the bot
bot.retrieve_from_database("What are the common issues with Dodo NBN reviews?")
bot.query("What are the common issues with Dodo NBN reviews?")
bot.retrieve_from_database("Can you find me a review that mentions the word 'speed'?")

