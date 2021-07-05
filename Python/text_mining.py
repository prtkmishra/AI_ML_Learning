"""
********************************************************************************************
file: text_mining.py
author: @Prateek Mishra
Description: Text Mining using Pandas Vectorization
********************************************************************************************
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import Counter
import requests
from pandas.core.common import flatten
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

start_time = time.time()
""""
********************************************************************************************
Import CSV training set into pandas
Define STOP Words
********************************************************************************************
"""
text = pd.read_csv('./data/Corona_NLP_train.csv',encoding='latin')
# STOP_WORDS = requests.get( "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt" ).content.decode('utf-8').split( "\n" )
# STOP_WORDS = open("./data/english-stop-words-large.txt",'r',encoding='utf-8').read().splitlines()
STOP_WORDS = ['a', "a's", 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 'against', "ain't", 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', "aren't", 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', "c'mon", "c's", 'came', 'can', "can't", 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', "couldn't", 'course', 'currently', 'd', 'definitely', 'described', 'despite', 'did', "didn't", 'different', 'do', 'does', "doesn't", 'doing', "don't", 'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', "hadn't", 'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', "he's", 'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', "let's", 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', 'should', "shouldn't", 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 't', "t's", 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that's", 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there's", 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', "wasn't", 'way', 'we', "we'd", "we'll", "we're", "we've", 'welcome', 'well', 'went', 'were', "weren't", 'what', "what's", 'whatever', 'when', 'whence', 'whenever', 'where', "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', "who's", 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', "won't", 'wonder', 'would', 'would', "wouldn't", 'x', 'y', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 'z', 'zero', 'a', ',', '.', '?', '!', '|', ':', "'", '"', ';', '<NUM>', '?', '$', 'km', 's', 'u', '&', '#', "'s", '/', 'dr.']

"""
********************************************************************************************
Define functions for vectorization
Inclides: 
    createListDF(): Converts DF of tweets into list of words and maps to a new DF
    removeSTOP(): Removes STOP words from input DF
    removeDUP: Remove duplicate words from a tweet
********************************************************************************************    
"""
def createListDF(df,question1_2=True):
    words = str(list(flatten(df))).split()
    df1 = pd.DataFrame({'words':words})
    df2 = df1['words'].str.replace('[\W_]','')
    if question1_2:
        check = Counter(df2).most_common(10)
        print("\nThe total number of all words (including repetitions): ",df2.count().sum(),"\nThe number of all distinct words :",df2.nunique(),"\n10 most frequent words in the corpus: ",check)
    return df2

def removeSTOP(df):
    L = df.loc[~df['split'].isin(STOP_WORDS)]
    X = L.loc[L['split'].str.len()>2].squeeze()
    return X

def removeDUP(string):
    s = string.tolist()
    a = s[0].split()
    mylist = list(set(a))
    return mylist

def classifier(text):
    """
    Input: Dataframe of Tweets and Sentiments
    Method: Count vectorizer to create WMatrix
            Fit the matrix in MultiNomialNB classisifer
            return predicted sentiments and score
    """
    vectorizer = CountVectorizer(analyzer='word',min_df=4,ngram_range=(1, 3),stop_words='english',strip_accents='ascii')
    corpus = text['OriginalTweet'].to_numpy()
    target = text['Sentiment'].to_numpy()
    WMatrix = vectorizer.fit_transform(corpus)
    clf = MultinomialNB()
    clf.fit(WMatrix, target)
    ypredicted=clf.predict(WMatrix)
    score=clf.score(WMatrix,target)
    print('\nClassification Error:',(1-score)*100,' %')

"""Convert tweets into lower form and remove all REGEX from the tweets"""
result = text['OriginalTweet'].str.lower().str.replace('[\W_]',' ')

"""
********************************************************************************************
Start question specific commands
********************************************************************************************
"""
print("********************************************************************************************\n\t\t\tLET'S BEGIN\n********************************************************************************************")
print("\n*********************** Question 1.1 ***********************")
"""Identify Unique elements in Sentiment attribute"""
_sentiments = text.Sentiment.unique()

"""Sort the Sentiment Attribute based on count of occurance"""
_second = text['Sentiment'].value_counts(sort=True,ascending=False)

"""Filter Date for Extremely Positive Tweet""" 
EPTweet = text.loc[text['Sentiment']=='Extremely Positive'].mode()

"""Print Question 1.1"""
print('Possible sentiments that a tweet may have',_sentiments.tolist(),'\nThe second most popular sentiment in the tweets is ',_second.index[1],' with ',_second.iloc[1],' Tweets','\nThe date with the greatest number of extremely positive tweets ',EPTweet['TweetAt'].iloc[0])

print("\n*********************** Question 1.2 ***********************")
"""Create List"""
newDF = pd.DataFrame({'split':createListDF(result)})
"""Remove STOP words"""
X = removeSTOP(newDF)
"""Identify 10 most common words"""
check1 = Counter(X).most_common(10)
"""Print Question 1.2"""
print("\nAfter cleanup:,\nThe total number of all words (including repetitions): ",X.count().sum(),"\nThe number of all distinct words :",X.nunique(),"\n10 most frequent words in the new corpus: ",check1)

print("\n*********************** Question 1.3 ***********************")
"""Create DF from result"""
withoutRE = pd.DataFrame({'rumba':result})
"""Use vectorization to remove duplicate words per tweet"""
withoutRE['rumba']= withoutRE.apply(lambda x: removeDUP(x), axis=1,raw=True)
"""Create a list of words and convert to DF"""
newDF1_4 = pd.DataFrame({'split':createListDF(withoutRE['rumba'],question1_2=False)})
"""Remove STOP words"""
X1_4 = removeSTOP(newDF1_4)
"""Find unique words count"""
Unique = X1_4.value_counts(normalize=True,ascending=True)

print("Saving plot as ./output/Question1_3.png")
Unique.plot()
plt.title('Words Proportion in Corpus')
plt.xlabel('words')
plt.ylabel('Proportion')
plt.savefig('./output/Question1_3.png')

print("\n*********************** Question 1.4 ***********************")
"""Take subset of original Dataframe"""
tweet = text[['OriginalTweet','Sentiment']]
"""Run Classifier"""
classifier(tweet)

print("\n*********************** End of run ***********************")
print("Total Time taken --- %s seconds ---" % (time.time() - start_time))