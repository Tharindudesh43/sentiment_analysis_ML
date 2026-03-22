import numpy as np
import pandas as pd
import re
import string
import pickle
import os

from nltk.stem import PorterStemmer
ps = PorterStemmer()

# Get base directory (VERY IMPORTANT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model_path = os.path.join(BASE_DIR, 'static', 'model', 'model.pickle')
with open(model_path, 'rb') as file:
    model = pickle.load(file)


# Load stopwords
stopwords_path = os.path.join(BASE_DIR, 'static', 'model', 'corpora', 'stopwords', 'english')
with open(stopwords_path, 'r') as file:
    sw = file.read().splitlines()

# Load vocabulary
vocab_path = os.path.join(BASE_DIR, 'static', 'model', 'vocabulary.txt')
vocab = pd.read_csv(vocab_path, header=None)
tokens = vocab[0].tolist()

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation,'')
    return text

def preprocessing(text):
    data = pd.DataFrame([text],columns=['tweet'])
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*','',x,flags=re.MULTILINE) for x in x.split()))
    data['tweet'] = data['tweet'].apply(remove_punctuations)
    data['tweet'] = data['tweet'].str.replace(r'\d+','',regex=True)
    data['tweet'] =  data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    data['tweet'] =  data['tweet'].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data['tweet']

def vectorization(ds):
    vectorized_list = []
    for sentence in ds:
        sentence_list = np.zeros(len(tokens))
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_list[i] = 1 
        vectorized_list.append(sentence_list)
    vectorized_list_new =  np.asarray(vectorized_list,dtype=np.float32)
    return vectorized_list_new

def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    if prediction == 1:
        return 'negative'
    else:
        return 'positive'