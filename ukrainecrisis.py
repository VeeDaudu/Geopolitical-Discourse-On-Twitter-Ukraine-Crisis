import tweepy 
import pandas as pd
import numpy as np
import csv 
import re 
import string 
import glob  
import requests 
import matplotlib.pyplot as plt
import concat

from collections import Counter

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer

import textblob
from textblob import TextBlob

from wordcloud import WordCloud
from emot.emo_unicode import UNICODE_EMOJI



# Access keys and codes from Twitter Developer Account
consumer_key = 'rEzxoqFLWLSFQBv1B5SsRz3ge'
consumer_secret = 'IRYNiiYHwypZbk5yp8DfIk9m4Ses7oYQOeMiASFmMLyMWjyG5C'
access_key= '1499372563457265674-FK1fTGLsWlB182jPEFvRlZ9gfRbtNH'
access_secret = 'gkPLvbD5zfsDsOxVVyzUtXGU52IKOFRcZO0qww4IXdU0m'

# Pass in twitter API authentication key
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret) 
api = tweepy.API(auth,wait_on_rate_limit=True)
sleep_on_rate_limit=False



# Timeframe
since_start= "2022-02-01"
since_end = "2022-07-24"


# Collect tweets using the Cursor object and scrape tweets individually:
def get_tweets(search_query, num_tweets):
    tweet_list = [tweets for tweets in tweepy.Cursor(api.search_tweets,
                                    q=search_query,
                                    lang="en",
                                    since_id = since_start,
                                    tweet_mode='extended').items(num_tweets)]
    for tweet in tweet_list:
        tweet_id = tweet.id # get user_id
        created_at = tweet.created_at # get time of tweet
        text = tweet.full_text # get the tweet
        location = tweet.user.location # get user's location
        retweet = tweet.retweet_count # get number of retweets
        favorite = tweet.favorite_count # get number of likes
        with open('Ukraine1.csv','a', newline='', encoding='utf-8') as csvFile:
            csv_writer = csv.writer(csvFile, delimiter=',') 
            csv_writer.writerow([tweet_id, created_at, text, location, retweet, favorite]) 
            
            


# Create keywords to search for, filter Links, retweets, replies.
search_words = "UkraineCrisis OR #UkraineRussiaWar OR #UkraineWar OR #ukraine"
search_query = search_words + " -filter:retweets AND -filter:replies"

#  Pass in search query and the number of tweets to retrieve
get_tweets(search_query,5000) 

            


