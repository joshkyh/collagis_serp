import datetime

import requests
import os
import json
import pandas as pd
from pprint import pprint
import datetime
from tqdm import tqdm

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")

search_url = "https://api.twitter.com/2/tweets/search/recent"

# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
#query_params = {'query': '(from:twitterdev -is:retweet) OR #twitterdev','tweet.fields': 'author_id'}


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

def connect_to_endpoint(url, params):
    response = requests.get(url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def recent_search(start_date:str, next_token:str = None):
    '''
    Function to get all reviews for a given data_id. Will call itself recursively if there is a next_page_token returned by api_response
    :param start_date:
    :param next_token:
    :return:
    '''

    #start_date = '2023-09-20'

    # Assert that start_date is of the format YYYY-MM-DD
    assert len(start_date) == 10, "start_date should be of the format YYYY-MM-DD"
    assert start_date[4] == '-' and start_date[7] == '-', "start_date should be of the format YYYY-MM-DD"


    start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')

    # end_date is one day after start_date
    end_date_obj = start_date_obj + datetime.timedelta(days=1)
    end_date = end_date_obj.strftime('%Y-%m-%d')

    # Function to get all reviews for a given start_time. Will call itself recursively if there is a next_page_token returned by api_response

    query_params = {'query': '@tabcomau -is:retweet',
                    'tweet.fields': 'author_id',
                    'user.fields': 'username',
                    'tweet.fields': 'created_at,author_id',
                    'start_time': f'{start_date}T00:00:00+11:00',
                    'end_time': f'{end_date}T00:00:00+11:00',
                    'max_results': '100'
                    }
    if next_token is not None:
        query_params['next_token']= next_token

    dict_response = connect_to_endpoint(search_url, query_params)

    column_names = ['author_id', 'id', 'created_at', 'text']
    results = pd.DataFrame(columns=column_names)

    new_data = pd.DataFrame(dict_response['data'])
    new_data = new_data.reindex(columns=column_names)

    # Convert datetime 2023-09-25T06:05:22.000Z to UTC+11
    new_data['created_at'] = pd.to_datetime(new_data['created_at']).dt.tz_convert('Australia/Sydney')

    # Row bind the new_data to the results dataframe
    results = pd.concat([results, new_data], ignore_index=True)


    if 'next_token' in dict_response['meta']:
        # If there is a next_token in the response, recursively call the function again
        next_token = dict_response['meta']['next_token']
        results = pd.concat([results, recent_search(start_date, next_token)], ignore_index=True)
        return results

    else:
        print('No next_token in response')
        return results


if __name__ == '__main__':

    # Create a list of dates from 6 days ago to yesterday
    date_list = []
    for i in range(1, 7):
        date_list.append((datetime.datetime.today() - datetime.timedelta(days=i)).strftime('%Y-%m-%d'))

    # Iterate through date_list
    for date in tqdm(date_list):
        results = recent_search(date)
        # export parquet to data
        results.to_parquet(f'data/twitter/{date}.parquet', index=False)


    # Load parquet files into a dataframe
    df = pd.concat([pd.read_parquet(f'data/twitter/{file}') for file in os.listdir('data/twitter/')], ignore_index=True)