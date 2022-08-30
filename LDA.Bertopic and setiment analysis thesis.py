# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 10:15:17 2022

@author: verao
"""

import pandas as pd
import json
from pprint import pprint
import pickle

world = pd.read_csv('C:/Users/verao/Desktop/topic modelling/Topic Modeling/world-cities.csv')
world[world['name'] == 'London']
world_countries = list(set(world['country'].tolist()))

df = pd.read_csv('C:/Users/verao/Desktop/topic modelling/Topic Modeling/data.csv', header=None)


df_countries = df[3].values.tolist()
new_names = []
for country in df_countries:
    if country == 'UK':
        new_names.append('United Kingdom')
    elif not type(country) == str:
        new_names.append('')
    elif country == 'USA':
        new_names.append('United States')
    elif len(country.split(' ')) > 2:
        new_names.append('')
    else:
        new_names.append(country)
        
 
countries_processed = []
for entry in pd.Series(new_names).value_counts().iteritems():
    if entry[1] > 5 and entry[0] != '':
        countries_processed.append(entry[0])        
        
        
        
#processed countries
countries_processed        



countries_map = {'London': 'United Kingdom', 
                'London, England': 'United Kingdom', 
                'Almere, Nederland': 'United Kingdom', 
                'England': 'United Kingdom', 
                'Russia': 'United Kingdom', 
                'Sweden': 'United Kingdom', 
                'EU ': 'United Kingdom', 
                'Ireland': 'United Kingdom', 
                'Marbella, Spain': 'United Kingdom', 
                'Scotland': 'United Kingdom', 
                'Europe': 'United Kingdom', 
                'Amsterdam': 'United Kingdom', 
                'Istanbul': 'United Kingdom', 
                'Berlin': 'United Kingdom', 
                'The Netherlands': 'United Kingdom', 
                'Germany': 'United Kingdom', 
                'Washington, DC': 'United States', 
                'Washington, D.C.': 'United States', 
                'U.S.A': 'United States', 
                'Chicago': 'United States', 
                'Florida, USA': 'United States', 
                'Early, TX': 'United States', 
                'California, USA': 'United States', 
                'Washington DC': 'United States', 
                ' USA': 'United States', 
                'New Jersey': 'United States', 
                'Hartford, CT': 'United States', 
                'Northeast U.S': 'United States', 
                'Україна': 'Ukraine',
                'Kyiv': 'Ukraine',
                'Kiev': 'Ukraine',
                'Kyiv, Ukraine': 'Ukraine',
                'Украина': 'Ukraine',
                'New York': 'United States',
                'United States': 'United States',
                'USA': 'United States',
                'UK': 'United Kingdom',
                'United Kingdom': 'United Kingdom',
                'London, UK': 'United Kingdom',
                'Everywhere': 'Everywhere',
                'Planet Earth': 'Everywhere',
                'Global': 'Everywhere',
                'Brussels': 'United Kingdom',
                'India': 'India',
                'Pakistan': 'India',
                'New Delhi': 'India',
                'Delhi, India': 'India',
                'Delhi, India': 'India',
                'Lagos, Nigeria': 'Africa',
                'Uganda': 'Africa',
                'Africa': 'Africa',
                'Mumbai, India': 'India'}




continents_map = {'United Kingdom': 'Europe',
                'United States': 'USA',
                'Ukraine' : 'Europe',
                'India' : 'Middle East',
                'Africa': 'Africa'
                }




df['countries'] = df[3].map(countries_map)
df['topics'] = df['countries'].map(continents_map)


df_clean_countries = df[df['topics'].notna()] 

df_clean_countries


#preprocess

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
from nltk.tokenize import word_tokenize
import preprocessor as p
from string import punctuation


p.clean(df[2].tolist()[0])


def most_frequent_words(tweets:list):
    count_vectorizer =  CountVectorizer(max_features=1000)
    feature_vector = count_vectorizer.fit(tweets)
    features = feature_vector.get_feature_names()
    train_ds_features = count_vectorizer.transform(tweets)
    features_counts =  np.sum(train_ds_features.toarray(),axis=0)
    features_counts = pd.DataFrame(dict(features = features, counts = features_counts))
    features_counts.sort_values('counts', ascending=False)[0:15]
    return features_counts['features'].values.tolist()


def remove_stopwords_and_numbers(tweet:str):
    tokens = word_tokenize(tweet)
    new_tweet = [word for word in tokens if word not in stopwords and not word.isnumeric() and word not in punctuation and len(word)>3]
    return ' '.join(new_tweet)



def remove_not_frequent(tweet:str, most_frequent:list):
    tokens = word_tokenize(tweet)
    new_tweet = [word for word in tokens if word in most_frequent]
    return ' '.join(new_tweet)

most_frequent = most_frequent_words(df[2].tolist())


tweets_clean = []
for tweet in df_clean_countries[2].tolist():
    tweet = p.clean(tweet)
    tweet_clean = remove_stopwords_and_numbers(tweet)
    # tweet_clean = remove_not_frequent(tweet_clean, most_frequent)
    # if tweet_clean:
    tweets_clean.append(tweet_clean)
    
    
df_clean_countries['tweet_clean'] = tweets_clean
df_clean_countries = df_clean_countries[df_clean_countries['tweet_clean'].notna()]

df_clean_countries.to_csv('C:/Users/verao/Desktop/topic modelling/Topic Modeling/clean_data.csv')


#topic modelling LDA
df = pd.read_csv('C:/Users/verao/Desktop/topic modelling/Topic Modeling/clean_data.csv')

import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim
from gensim.models import CoherenceModel


# Sentence to word function
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence).lower(), deacc=True))
        
        
data_words = list(sent_to_words(df['tweet_clean']))        
data_words[0]



# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])



countries = list(set(df['topics'].tolist()))
countries


coherence_lda_df = []
for country in countries:
    df_country = df[df['topics'] == country]
    data_words = list(sent_to_words(df_country['tweet_clean']))
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    # print(corpus[:1][0][:30])
    # number of topics
    # num_topics = 8
    # Build LDA model
    for num_topics in range(6, 12):
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics)
        # Print the Keyword in the 10 topics
        pprint(lda_model.print_topics())
        doc_lda = lda_model[corpus]
        # Visualize the topics
        pyLDAvis.enable_notebook()
        LDAvis_data_filepath = f'./topic_modeling/{country}'
        # # this is a bit time consuming - make the if statement True
        # # if you want to execute visualization prep yourself
        if 1 == 1:
            LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
            with open(LDAvis_data_filepath, 'wb') as f:
                pickle.dump(LDAvis_prepared, f)
        # load the pre-prepared pyLDAvis data from disk
        with open(LDAvis_data_filepath, 'rb') as f:
            LDAvis_prepared = pickle.load(f)
        pyLDAvis.save_html(LDAvis_prepared, f'./topic_modeling_viz/{country}_topics_{num_topics}.html')

        coherence_model_lda = CoherenceModel(model=lda_model, 
                                        texts=texts, 
                                        dictionary=id2word, 
                                        coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        coherence_lda_df.append([num_topics, coherence_lda, country])
        LDAvis_prepared
        
coherence_lda_df_pd = pd.DataFrame(coherence_lda_df, columns=['Number of topics', 'Coherence', 'Topic'])
coherence_lda_df_pd

countries
for county in countries:
    print(county)
    vizualize_coherence = coherence_lda_df_pd[coherence_lda_df_pd['Topic'] == county]
    vizualize_coherence.plot.line(x='Number of topics', y='Coherence', rot=0, title=county)
    
    
#Bertopic
from bertopic import BERTopic

docs = []
for country in countries:
    df_country = df[df['topics'] == country]
    df_country = df_country['tweet_clean'].tolist()
    df_country = [text.lower() for text in df_country if type(text) == str]
    docs.extend(df_country)


topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
topics, probs = topic_model.fit_transform(docs)

freq = topic_model.get_topic_info(); freq.head(5)

topic_model.visualize_topics()

topic_model.visualize_barchart(top_n_topics=5)


#sentiment Analysis
from textblob import TextBlob

def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score < -0.1:
        return 'Negative'
    elif score > 0.1:
        return 'Positive'
    else:
        return 'Neutral'
    
df = df[df['tweet_clean'].notna()]


sent = []
for tweet in df['tweet_clean'].tolist():
    # print(tweet)
    sent.append(get_sentiment(tweet))
    
    
df['sentiment'] = sent

import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(10, 8))
for country in countries:
    df_process = df[df['countries'] == country]
    dfg = df_process.groupby(['sentiment'])['tweet_clean'].count()
    # Plot
    # print(dfg)
    # if not dfg.empty:
    dfg.plot(kind='bar', title=f'{country}', ylabel='Sentiment', xlabel='Count', figsize=(6, 5))
    plt.show()

       