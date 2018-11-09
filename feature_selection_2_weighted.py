"""
Case 2.b
Feature Selection 2 + Tfidf Weighting
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
s_ne_data = s_ne[['text' ]]
X_raw = s_ne_data.values

# load target values
s_ne_target = s_ne[['class']]
y = s_ne_target.values
c, r = y.shape
y = y.reshape(c,)

# ------------------------------------------------------------------#

from nltk.tokenize import WordPunctTokenizer
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Defining data cleaning function
tok = WordPunctTokenizer()

pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
pat3 = r'#[A-Za-z0-9]+'
combined_pat = r'|'.join((pat1, pat2, pat3))


def tweet_cleaner(text):
    """
    This function preps the tweets for feature extraction.
    Parses the html, decodes the UTF8, removes URLs, and lowercases all the text.
    Hashtags (#) and mentions (@) are left.
    
    params: tweet bytes object
    returns list of cleaned tweet
    """
    
    # html decoding
    soup = BeautifulSoup(text, 'html.parser')
    souped = soup.get_text()
    # remove URLs
    stripped = re.sub(combined_pat, '', souped)
    # decode UTF8
    try:
        clean = stripped.decode("utf8").encode('ascii','ignore')
    except:
        clean = stripped
     # remove non-alphabetic characters (#, numbers, and punctuation)
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    # normalize to lowercase
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()
    #return words


def remove_stopwords(text):
    sw = set(stopwords.words('english'))
    words = word_tokenize(text)
    words_filtered = []
    
    for w in words:
        if ((w not in sw) and (len(w) > 2)): #remove stopwords and lone letters
            words_filtered.append(w)
    
    return words_filtered
#return content


X_cleaned = []
for t in range(len(X_raw)):
    X_cleaned.append(remove_stopwords(tweet_cleaner(X_raw[t][0])))
print("Finished cleaning + feature selection.")


X_almost_clean = []
g = ''

for i in X_cleaned: 
    for j in i: 
        g = g + ' ' + j # add the words of one tweet together, then append that string to flat list
    X_almost_clean.append(g)
    g = ''
    

# begin stemmer code

from nltk.stem import SnowballStemmer


def stemmatize(text):
    sb = SnowballStemmer("english")
    return sb.stem(text)

X_clean = []
for t in X_almost_clean:
    X_clean.append(stemmatize(t))



# ---------------------- TRAINING & TESTING ---------------------------- #

text_train, text_test, y_train, y_test = train_test_split(X_clean, y, random_state=0)


# import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# create transform object
tfidf_vectorizer = TfidfVectorizer(lowercase=False).fit(text_train)
X_train = tfidf_vectorizer.transform(text_train)

X_test = tfidf_vectorizer.transform(text_test)


    
# import libaries
from sklearn.tree import DecisionTreeClassifier

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