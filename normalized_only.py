"""
Case 0.a
Normalized Only
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# load Southern and New England data sets
south = pd.read_csv("south.csv")
new_england = pd.read_csv("new_england.csv")


# reduce data set size
south_rest, south_10k = train_test_split(south, test_size=10000, random_state=0)
new_england_rest, new_england_10k = train_test_split(new_england, test_size=10000, random_state=0)

# combine south_10k + west_10k into the data set
frames = [south_10k, new_england_10k]
s_ne = pd.concat(frames)


# get text
s_ne_data = s_ne[['text']]
X_raw = s_ne_data.values

# load target values
s_ne_target = s_ne[['class']]
y = s_ne_target.values
c, r = y.shape
y = y.reshape(c,)



from nltk.tokenize import WordPunctTokenizer
import re
from bs4 import BeautifulSoup


# Defining data cleaning function
tok = WordPunctTokenizer()
pattern = r'https?://[A-Za-z0-9./]+'
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )""" 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
] 
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def tweet_cleaner(text):
    """
    This function preps the tweets for feature extraction.
    Parses the html, decodes the UTF8, removes URLs, removes numbers and punctuation, and lowercases all the text.
    Hashtags (#) and mentions (@) are left, could be extracted as features in next step.
    
    params: tweet bytes object
    returns cleaned tweet
    """
    
    # html decoding
    soup = BeautifulSoup(text, 'html.parser')
    souped = soup.get_text()
    # remove URLs
    stripped = re.sub(pattern, '', souped)
    # decode UTF8
    try:
        clean = stripped.decode("utf8").encode('ascii','ignore')
    except:
        clean = stripped
    # tokenize and preprocess
    result = preprocess(clean, lowercase=True)
    #return result
    return (" ".join(result)).strip()


X_clean = []
for t in range(len(X_raw)):
    X_clean.append(tweet_cleaner(X_raw[t][0]))
print("Finished cleaning.")



# ---------------------- TRAINING & TESTING ---------------------------- #


# import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

text_train, text_test, y_train, y_test = train_test_split(X_clean, y, random_state=0) 

count_vectorizer = CountVectorizer(lowercase=False).fit(text_train)
X_train = count_vectorizer.transform(text_train)

X_test = count_vectorizer.transform(text_test)

# import libaries
from sklearn.tree import DecisionTreeClassifier

#get set to fit classifier
#X_train, X_test, y_train, y_test = train_test_split(count_vector, y, random_state=0)

# initialize classifier
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)
prediction = dt.predict(X_test)
print("Finished predictions.")


# ---------------------- Model Performance Measures --------------------- #

#function below returns sensitivity value
def rec(tp, fp, tn, fn):
    return tp / (tp+fn) 
#function below returns accuracy
def acc(tp, fp, tn, fn):
    return (tp+tn) / (tn+fp+fn+tp)
#function below returns precision
def prec(tp, fp, tn, fn):
    return tp/(tp + fp)
#function below returns F1-Score
def f1_score(prec, sens):
    return (2 * prec * sens)/(prec + sens) 

# Positive class = successful = 1
# Negative class = failed = 0
def model_perf_measure(y_actual, y_pred):
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for i in range(len(y_pred)): 
        if (y_actual[i]==1) and (y_pred[i]==1):
           TP += 1
        if (y_actual[i]==1) and (y_pred[i]==0):
           FN += 1
        if (y_actual[i]==0) and (y_pred[i]==1):
           FP += 1
        if (y_actual[i]==0) and (y_pred[i]==0):
           TN += 1
    
    recall = rec(TP, FP, TN, FN)
    accuracy = acc(TP, FP, TN, FN)
    precision = prec(TP, FP, TN, FN)
    f1 = f1_score(precision, recall)
    perf = ['TP', 'FP', 'TN', 'FN', 'accuracy', 'recall', 'precision','f1']
        
    return(np.stack((perf, [TP, FP, TN, FN, accuracy, recall, precision, f1])))


perf = model_perf_measure(y_test, prediction)
print(perf.T)
