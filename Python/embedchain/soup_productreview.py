import json
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import random

# Chrome options
chrome_options = Options()
chrome_options.add_argument(
    "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.3")

# Initialize Selenium webdriver
driver = webdriver.Chrome(options=chrome_options)

# Navigate to the webpage
reviews = []
page_num=10
for page_num in tqdm(range(10, 15)):
    url = f"https://www.productreview.com.au/listings/dodo?page={page_num}#reviews-list"
    driver.get(url)

    # Wait for some time for the page to load
    time.sleep(random.uniform(5.5, 10.5))

    # Scroll the webpage to emulate user behavior
    for i in range(int(random.uniform(2, 5))):

        driver.execute_script(f"window.scrollBy(0, {int(random.uniform(100, 500))});")
        time.sleep(random.uniform(0.5, 1.5))

    # Now you can extract the page content and parse it with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Get page source
    page_source = driver.page_source

    # Parse HTML with Beautiful Soup
    soup = BeautifulSoup(page_source, 'html.parser')

    # Search for the script tag with the type "application/ld+json"
    script_tag = soup.find('script', {'type': 'application/ld+json'})


    # If found, proceed to extract and parse the JSON-LD data

    if script_tag:
        json_ld_str = script_tag.string.strip()
        json_ld_dict = json.loads(json_ld_str)

        # Now you can navigate through the dictionary to find the reviewBody
        for review in json_ld_dict.get('review', []):
            review_body = review.get('reviewBody', 'N/A')
            if review_body != 'N/A':
                reviews.append(review_body)

    # Wait 1 second
    time.sleep(1)

# Rename column as snippet
dfr = pd.DataFrame(reviews, columns=['snippet'])

# Save parquet file
dfr.to_parquet('data/productreviews/dodo_10-14.parquet', index=False)


# Close the browser
driver.close()
