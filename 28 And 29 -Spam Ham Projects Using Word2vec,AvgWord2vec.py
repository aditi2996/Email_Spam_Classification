#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().system('pip install gensim')


# In[1]:


import gensim
from gensim.models import Word2Vec, KeyedVectors


# In[2]:


import gensim.downloader as api

wv = api.load('word2vec-google-news-300')

vec_king = wv['king']


# In[3]:


vec_king


# In[5]:


import pandas as pd
messages=pd.read_csv('SMSSpamCollection.txt',
                    sep='\t',names=["label","message"])


# In[6]:


messages.shape


# In[7]:


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


# In[8]:


import re
import nltk
nltk.download('stopwords')


# In[9]:


corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [lemmatizer.lemmatize(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)


# In[10]:


[[i,j,k] for i,j,k in zip(list(map(len,corpus)),corpus, messages['message']) if i<1]


# In[11]:


corpus


# In[12]:


from nltk import sent_tokenize
from gensim.utils import simple_preprocess


# In[13]:


words=[]
for sent in corpus:
    sent_token=sent_tokenize(sent)
    for sent in sent_token:
        words.append(simple_preprocess(sent))


# In[14]:


words


# In[15]:


import gensim


# In[16]:


## Lets train Word2vec from scratch
model=gensim.models.Word2Vec(words)


# In[17]:


## To Get All the Vocabulary
model.wv.index_to_key


# In[18]:


model.corpus_count


# In[19]:


model.epochs


# In[20]:


model.wv.similar_by_word('good')


# In[21]:


model.wv['good'].shape


# In[22]:


words[0]


# In[23]:


def avg_word2vec(doc):
    # remove out-of-vocabulary words
    #sent = [word for word in doc if word in model.wv.index_to_key]
    #print(sent)
    
    return np.mean([model.wv[word] for word in doc if word in model.wv.index_to_key],axis=0)
                #or [np.zeros(len(model.wv.index_to_key))], axis=0)


# In[24]:


get_ipython().system('pip install tqdm')


# In[25]:


from tqdm import tqdm


# In[26]:


#apply for the entire sentences
import numpy as np
X=[]
for i in tqdm(range(len(words))):
    X.append(avg_word2vec(words[i]))


# In[27]:


len(X)


# In[42]:


X


# In[47]:


##independent Features
X_new=X


# In[48]:


messages.shape


# In[49]:


X[1]


# In[56]:


X_new[0].shape


# In[57]:


## Dependent Features
## Output Features
y = messages[list(map(lambda x: len(x)>0 ,corpus))]
y=pd.get_dummies(y['label'])
y=y.iloc[:,0].values


# In[58]:


y.shape


# In[59]:


X[0].reshape(1,-1).shape


# In[62]:


## this is the final independent features
# df=pd.DataFrame()
# for i in range(0,len(X)):
#     df=df.append(pd.DataFrame(X[i].reshape(1,-1)),ignore_index=True)
    


# In[61]:


import pandas as pd

df = pd.DataFrame()
for i in range(0, len(X)):
    # Reshape X[i] and create a DataFrame for each, then concatenate to df
    df = pd.concat([df, pd.DataFrame(X[i].reshape(1, -1))], ignore_index=True)


# In[63]:


df.head()


# In[64]:


df['Output']=y


# In[65]:


df.head()


# In[66]:


df.dropna(inplace=True)


# In[67]:


df.isnull().sum()


# In[83]:


## Independent Feature
X = df.iloc[:, :-1]


# In[84]:


X.isnull().sum()


# In[85]:


y=df['Output']


# In[86]:


## Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)


# In[87]:


X_train.head()


# In[88]:


X_train.dtypes


# In[89]:


y_train


# In[90]:


y_train = y_train.apply(lambda x: 1 if x == True else 0)


# In[91]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()


# In[92]:


classifier.fit(X_train,y_train)


# In[93]:


y_pred=classifier.predict(X_test)


# In[94]:


from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_test,y_pred))


# In[95]:


print(classification_report(y_test,y_pred))

