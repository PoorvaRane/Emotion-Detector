
# coding: utf-8

# In[2]:

from __future__ import division
import nltk 
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import *
from textblob.classifiers import NaiveBayesClassifier
from sklearn.cross_validation import KFold
from nltk.classify.naivebayes import NaiveBayesClassifier
from llda import LLDA
from word_prob_dist import word_distribution
from optparse import OptionParser


# In[3]:

'''
Reading the Dataset (ISEAR Dataset)
'''
Data = pd.read_csv('my_table.csv',header=None)
'''
36 - Class Label
40 - Sentence
'''


# In[4]:

'''
Emotion Labels
'''
emotion_labels = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt']


# In[5]:

'''
Negation words
'''
negation_words = ['not', 'neither', 'nor', 'but', 'however', 'although', 'nonetheless', 'despite', 'except', 'even though', 'yet']


# In[6]:

'''
Returns a list of all corresponding class labels
'''
def class_labels(emotions):
    labels = []
    labelset = []
    for e in emotions:
        labels.append(e)
        labelset.append([e])
    return labels, labelset


# In[7]:

'''
Removes unnecessary characters from sentences
'''
def removal(sentences):
    sentence_list = []
    count = 0
    for sen in sentences:
        count += 1
#         print count
#         print sen
#         print type(sen)
        s = nltk.word_tokenize(sen)
        characters = ["á", "\xc3", "\xa1", "\n", ",", "."]
        new = ' '.join([i for i in s if not [e for e in characters if e in i]])
        sentence_list.append(new)
    return sentence_list


# In[8]:

'''
POS-TAGGER, returns NAVA words
'''
def pos_tag(sentences):
    tags = [] #have the pos tag included
    nava_sen = []
    for s in sentences:
        s_token = nltk.word_tokenize(s)
        pt = nltk.pos_tag(s_token)
        nava = []
        nava_words = []
        for t in pt:
            if t[1].startswith('NN') or t[1].startswith('JJ') or t[1].startswith('VB') or t[1].startswith('RB'):
                nava.append(t)
                nava_words.append(t[0])
        tags.append(nava)
        nava_sen.append(nava_words)
    return tags, nava_sen


# In[9]:

'''
Performs stemming
'''
def stemming(sentences):
    sentence_list = []
    sen_string = []
    sen_token = []
    stemmer = PorterStemmer()
#     i = 0
    for sen in sentences:
#         print i,
#         i += 1
        st = ""
        for word in sen:
            word_l = word.lower()
            if len(word_l) >= 3:
                st += stemmer.stem(word_l) + " "
        sen_string.append(st)
        w_set = nltk.word_tokenize(st)
        sen_token.append(w_set)
        w_text = nltk.Text(w_set)
        sentence_list.append(w_text)
    return sentence_list, sen_string, sen_token


# In[10]:

'''
Write to file
'''
def write_to_file(filename, text):
    o = open(filename,'w')
    o.write(str(text))
    o.close()


# In[11]:

'''
Creating the dataframe
'''
def create_frame(Data):
    emotions = Data[36]
    sit = Data[40]
    labels, labelset = class_labels(emotions[1:])
    sent = removal(sit[1:])
    nava, sent_pt = pos_tag(sent)
    sentences, sen_string, sen_token = stemming(sent_pt)
    frame = pd.DataFrame({0 : labels,
                          1 : sentences,
                          2 : sen_string,
                          3 : sen_token,
                          4 : labelset})
    return frame


# In[12]:

c = create_frame(Data)


# In[20]:

'''
Reads the emotion representative words file
'''
def readfile(filename):
    f = open(filename,'r')
    representative_words = []
    for line in f.readlines():
        characters = ["\n", " ", "\r", "\t"]
        new = ''.join([i for i in line if not [e for e in characters if e in i]])
        representative_words.append(new)
    return representative_words


# In[21]:

'''
Makes a list of all words semantically related to an emotion and Stemming
'''
def affect_wordlist(words):
    affect_words = []
    stemmer = PorterStemmer()
    for w in words:
        w_l = w.lower()
        word_stem = stemmer.stem(w_l)
        if word_stem not in affect_words:
            affect_words.append(word_stem)
    return affect_words


# In[22]:

'''
Creating an emotion wordnet
'''
def emotion_word_set(emotions):
    word_set = {}
    for e in emotions:
        representative_words = readfile(e)
        wordlist = affect_wordlist(representative_words)
        word_set[e] = wordlist
    return word_set


# In[23]:

'''
Lexicon based approach - Check for lexicons
'''
def lexicon_based(sentences, word_set):
    text_vector = []
    for sen in sentences:
        s_vector = []
        for word in sen:
            w_vector = {}
            for emo in word_set:
                if word in word_set[emo]:
#                     print word
                    try:
                        if emo not in w_vector[word]:
                            w_vector[word].append(emo)
                    except KeyError:
                        w_vector[word] = [emo]
            if w_vector:
                s_vector.append(w_vector)
        if not s_vector:
            text_vector.append(s_vector)
        else:
            text_vector.append(s_vector)
    return text_vector


# In[24]:

'''
Lexicon based approach - Classify based on lexicons
'''
def classify_lexicon(text_vector, labels, emotion_labels):
    count = 0
    total = 0
    for j in range(len(text_vector)):
        sen = text_vector[j]
        sen_emo = np.empty(len(emotion_labels))
        sen_emo.fill(0)
        if sen:
            total += 1
            w_emo = []
            for word in sen:
                emotions =  word.values()[0][0]
#                 print emotions, type(emotions), j
                w_emo.append(emotions)
                i = emotion_labels.index(emotions)
                sen_emo[i] += 1
#             print sen_emo
            winner = np.argwhere(sen_emo == np.amax(sen_emo))
            indices = winner.flatten().tolist()
            for i in indices:
                if emotion_labels[i] == labels[j]:
                    count += 1
                    break
#                 else:
#                     print j, text_vector[j]
    accuracy = count/len(text_vector)
    tot_accuracy = count/total
    return accuracy, tot_accuracy


# In[25]:

e = emotion_word_set(emotion_labels)
l = lexicon_based(c[1],e) 
a, b = classify_lexicon(l, c[0], emotion_labels)


# In[26]:

'''
Calculate pmi
'''
def pmi(x, y, sentences):
    count_x = 1
    count_y = 1
    count_xy = 1
    for sen in sentences:
        if x and y in sentences:
            count_xy += 1
            count_x += 1
            count_y += 1
        if x in sentences:
            count_x += 1
        if y in sentences:
            count_y += 1
        result = count_xy/(count_x * count_y)
    return result


# In[27]:

print a*100, '%'
print b*100, "%"


# In[ ]:


# In[20]:

'''
Getting synonyms from wordnet synsets
'''
from nltk.corpus import wordnet as wn
jw = wn.synsets('shame')
for s in jw:
    v = s.name()
    print wn.synset(v).lemma_names()


# In[28]:

'''
Creating training/testing set for Naive Bayes classifier TextBlob
'''
def create_dataset_textblob(sentences, emotions):
    train = []
    sen = []
    emo = []
    for s in sentences:
        sen.append(s)
    for e in emotions:
        emo.append(e)
    for i in range(len(sen)):
        s = sen[i]
        e = emo[i]
        train.append((str(s), e))
    return train


# In[29]:

'''
Testing for Naive Bayes Classifier
'''
def testing(cl, test):
    print cl.classify('angry')
    for s, e in test:
        r = cl.classify(s)
        print s, e, r
        if r == e:
            print "*"


# In[30]:

'''
Create dataset for nltk Naive Bayes
'''
def create_data(sentence, emotion):
    data = []
    for i in range(len(sentence)):
        sen = []
        for s in sentence[i]:
            sen.append(str(s))
        emo = emotion[i]
        data.append((sen, emo))
    return data


# In[31]:

'''
Get all words in dataset
'''
def get_words_in_dataset(dataset):
    all_words = []
    for (words, sentiment) in dataset:
        all_words.extend(words)
    return all_words


# In[32]:

'''
Getting frequency dist of words
'''
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


# In[33]:

'''
Extacting features
'''
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


# In[34]:

'''
Create test data
'''
def create_test(sentence, emotion):
    data = []
    sen = []
    emo = []
    for s in sentence:
        sen.append(str(s))
    for e in emotion:
        emo.append(e)
    for i in range(len(sen)):
        temp = []
        temp.append(sen[i])
        temp.append(emo[i])
        data.append(temp)
    return data


# In[35]:

'''
Classifier
'''
def classify_dataset(data):
    return         classifier.classify(extract_features(nltk.word_tokenize(data)))


# In[36]:

'''
Get accuracy
'''
def get_accuracy(test_data, classifier):
    total = accuracy = float(len(test_data))
    for data in test_data:
        if classify_dataset(data[0]) != data[1]:
            accuracy -= 1
    print('Total accuracy: %f%% (%d/20).' % (accuracy / total * 100, accuracy))


# # In[37]:

# # Create training and testing data
# sen = c[3]
# emo = c[0]
# l = len(c[3])
# limit = (9*l)//10
# sente = c[2]
# Data = create_data(sen[:limit], emo[:limit])
# test_data = create_test(sente[limit:], emo[limit:]) 


# # In[38]:

# # extract the word features out from the training data
# word_features = get_word_features(                    get_words_in_dataset(Data))


# # In[39]:

# # get the training set and train the Naive Bayes Classifier
# training_set = nltk.classify.util.apply_features(extract_features, Data)
# classifier = NaiveBayesClassifier.train(training_set)


# # In[40]:

# get_accuracy(test_data, classifier)


# In[19]:

b =  word_distribution(emotion_labels,c[1],c[0])
o = open('emotion_words.txt','w')
o.write(str(b))
o.close()


# In[ ]:



