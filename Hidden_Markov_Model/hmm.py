## Hidden Markov Model
##### Feb 4th 2023
##### William Lu
import pandas as pd
import numpy as np
import nltk
import re
import csv
from bs4 import BeautifulSoup
from _collections import defaultdict
import json
from sklearn.metrics import accuracy_score

path  = '/Users/William/Downloads/natural-language-processing/Hidden_Markov_Model/data/train'
df = pd.read_csv(path,on_bad_lines='skip',sep='\t',header= None)

# rename the column name 
new_df = df.copy()
new_dff = df.copy()
new_dfff = df.copy()
new_df['Occurance']=new_df.groupby([1])[2].transform(len)
new_df.loc[new_df.Occurance<3,1]="<unk>"
new_df['occurance']=new_df.groupby([1])[2].transform(len)

# rename the column
new_df.rename(columns={1: "word"},inplace=True)
# drop duplicate rows based on column word (making sure just one <unk>)
new_df.drop_duplicates(subset=['word'],inplace=True)
# we store one df that contains the tag for each word
tag_df = new_df.copy()
# drop unessary column, includinh 0, 2, and Occurance
new_df.drop(columns=[0, 2,"Occurance"],inplace=True)
# sort the dataframe by occurance
new_df.sort_values(by=['occurance'],ascending=False,inplace=True)

new_df.reset_index(inplace=True)
new_df.drop(columns=['index'],inplace=True)
# put <unk> to be on the first row of the dataframe
new_df.iloc[3], new_df.iloc[0] = new_df.iloc[0], new_df.iloc[3]
# resort it to make sure <unk> has index 0
new_df.reset_index(inplace=True)
# make sure the first few words have the correct order
new_df.iloc[3], new_df.iloc[1] = new_df.iloc[1], new_df.iloc[3]
new_df.reset_index(inplace=True)
# new_df

new_df.drop(columns=["level_0"],inplace=True)
# same but it's for word "."
new_df.iloc[3], new_df.iloc[2] = new_df.iloc[2], new_df.iloc[3]
new_df['index'] = new_df.index

# shift column word to be the first column
first_column = new_df.pop('word')
new_df.insert(0, 'word', first_column)
new_df

new_df.to_csv("vocab_csv.csv",index=False)
np.savetxt('/Users/William/Downloads/natural-language-processing/Hidden_Markov_Model/data/vocab.txt', new_df, delimiter=r'\t ', header=r'\t '.join(new_df.columns.values), fmt='%s', comments='', encoding=None)
print("success")
print("Selected threshold for unk words replace ment is 3.")
print("Total size of my vocabulary is 16920 after counting.")
print("There are 32537 <unk> after replacement.")


### Model Learning
new_dfff['Occurance']=new_dfff.groupby([1])[2].transform(len)
new_dfff.loc[new_dfff.Occurance<3,1]="<unk>"
new_dfff['occurance']=new_dfff.groupby([1])[2].transform(len)

# we want to grab the frequent word into learning HMM
new_dfff.drop_duplicates(subset=[1],inplace=True)
frequent = new_dfff.loc[new_dfff["Occurance"]>=3][1]

# define dictionaries
transition = {}
emission = {}
tag_count = {}
tag_tag = {}
tag_word = {}
new_dff['Occurance']=new_dff.groupby([1])[2].transform(len)
new_dff.loc[new_dff.Occurance<3,1]="<unk>"
new_dff['occurance']=new_dff.groupby([1])[2].transform(len)
# new_dff[1]

existed_keys = new_df['word'].tolist()
words = new_dff[1].values.tolist()
indices = df[0].values.tolist()
poss = df[2].values.tolist()
for w in range(len(words)-1):
    if(indices[w]<indices[w+1]):
        # we use ~ here because some tags have , or . as tag
        if '('+str(poss[w+1]) + '~' + str(poss[w])+')' in tag_tag:
            tag_tag['('+str(poss[w+1]) + '~' + str(poss[w])+')'] +=1
        else:
            tag_tag['('+str(poss[w+1]) + '~' + str(poss[w])+')'] =1
##### new_dff
for w in range(len(words)-1):
    if(indices[w]<indices[w+1]):
    # check e now
        if '('+str(words[w]) + '~' + str(poss[w])+')'in tag_word:
            tag_word['('+str(words[w]) + '~' + str(poss[w])+')'] +=1
        else:
            tag_word['('+str(words[w]) + '~' + str(poss[w])+')'] =1

# setting up start tag 
words_length = len(words)
for i in range (words_length):
    # if index is 1, it means that word is the start of the sentence
    if indices[i]==1:
        if '('+str(poss[i]) + '~' + '<s>'+')' in tag_tag:
            tag_tag['('+str(poss[i]) + '~' + '<s>'+')'] +=1
        else:
            tag_tag['('+str(poss[i]) + '~' + '<s>'+')'] =1
# tag_tag
# count the number of the <s> tag
tag_count['<s>']=0
for i in indices:
    if (i==1):
        # start of the sentence
        tag_count['<s>']+=1
# tag_count

# count other tag number to be in the denominator
for i in poss:
    if i not in tag_count:
        tag_count[i] = 1
    else:
        tag_count[i] +=1
print("the number of tags is ",len(tag_count))

# finding transition and emission
for i in tag_tag:
    ix = i.split('~')[1][:-1] # we need to exclude the right bracket ')'
    transition[i] = tag_tag[i]/tag_count[ix]   

for i in tag_word:
    ix = i.split('~')[1][:-1]
    emission[i] = tag_word[i]/tag_count[ix]

emission_transition = [emission, transition]
with open('/Users/William/Downloads/natural-language-processing/Hidden_Markov_Model/data/hmm.json', 'w') as output:
    json.dump(emission_transition, output)

print("The number of parameters for transition is:", len(transition))
print("The number of parameters for emission is:",len(emission))


dev = pd.read_csv('/Users/William/Downloads/natural-language-processing/Hidden_Markov_Model/data/dev',sep='\t',names=["index","word","tag"])
#dev

dev_word = dev["word"].values.tolist()
dev_index = dev["index"].values.tolist()
dev_tag = dev["tag"].values.tolist()
tag_list = list(tag_count.keys())
wordss = []
tagss= []
dev_word_true = []
dev_tag_true = []

# separte words in to a list, same for tags
for i in range(len(dev)-1):
    if dev_index[i]< dev_index[i+1]:
        wordss.append(dev_word[i])
        tagss.append(dev_tag[i])
    else:
        wordss.append(dev_word[i])
        dev_word_true.append(wordss)
        wordss =[]
        tagss.append(dev_tag[i])
        dev_tag_true.append(tagss)
        tagss=[]
# make frequent words to a list
frequent=list(frequent)

def greedy(stnce):
    tag=[]
    prb=0
    pos = 'unk'
    # check the start of the sentence is unk or not
    if stnce[0] not in frequent:
        stnce[0]='<unk>'
        
    for i in tag_list:
        # some index may not exist so need to use try statement
        # reference: https://www.w3schools.com/python/python_try_except.asp
        try:
            a = transition['('+i+'~'+'<s>'+')']*emission['('+stnce[0]+'~'+i+')']
            if a > prb:
                prb = a
                pos = i
        except:
            pass
    tag.append(pos)
    #check if the word is in the vocabulary
    for i in range(1,len(stnce)):
        if stnce[i] not in frequent:
            stnce[i] = "<unk>"
        prb2 = 0
        pos2 ='unk'
        for j in tag_list:
            # some index may not exist so need to use try statement
            try:
                b = transition['('+j+'~'+tag[-1]+')'] * emission['('+stnce[i]+'~'+j+')']
                if b >prb2:
                    prb2 = b
                    pos2 = j
            except:
                pass
        tag.append(pos2)
    return tag

tag_pred=[]
for i in dev_word_true:
    tag_pred.append(greedy(i))
tag_pred= sum(tag_pred, [])
tag_acc = sum(dev_tag_true, [])

accuracy = accuracy_score(tag_acc, tag_pred)
print("The accuracy score for greedy decoding HMM is:",accuracy)

test = pd.read_csv('/Users/William/Downloads/natural-language-processing/Hidden_Markov_Model/data/test',sep='\t',names=["index","word"])

word_test_data = []
test_words=test["word"].values.tolist()
test_index = test["index"].values.tolist()
test_words_true=[]
for i in range(len(test)-1):
    if test_index[i]< test_index[i+1]:
        word_test_data.append(test_words[i])
    else:
        word_test_data.append(test_words[i])
        test_words_true.append(word_test_data)
        word_test_data =[]

tag_pred_test =[]
for i in test_words_true:
    tag_pred_test.append(greedy(i))
tag_pred_test

# make the word and tags into list of tuples
wl = []
for t in test_words_true:
    q= 1
    for i in t:
        wl.append((i,q))
        q+=1
tl =[]
for t in tag_pred_test:
    q= 1
    for i in t:
        tl.append((i,q))
        q+=1

wll= pd.DataFrame(wl,columns = ["word","index"])
tll = pd.DataFrame(tl,columns = ["tag","index2"])

greedy_df = pd.concat([wll,tll],axis=1)
greedy_df = greedy_df.drop('index2', axis=1)
first_column = greedy_df.pop('index')
greedy_df.insert(0, 'index', first_column)
#greedy_df

greedy_df.to_csv("greedy.csv",index=False)
np.savetxt('/Users/William/Downloads/natural-language-processing/Hidden_Markov_Model/data/greedy.out', greedy_df, delimiter=r'\t ', header=r'\t '.join(greedy_df.columns.values), fmt='%s', comments='', encoding=None)
print("success")


def viterbi(stnce):
    if stnce[0] not in frequent:
        stnce[0] = '<unk>'
    # create dictionaries to store cumulative prbabilities of a position of a tag
    TAG = {}
    # create a dictionary to know which tag leads to the max cumulative probability
    last_TAG = {}
    for i in range(len(stnce)):
        TAG[i] = {}
        last_TAG[i] = {}
        
    # for the first position cumulative probability 
    for t in tag_list:
        if '('+t + '~' + '<s>'+')' in transition:
            try:
                TAG[0][t] = transition['('+t + '~' + '<s>'+')'] * emission['('+stnce[0] + '~' + t+')']
            except:
                TAG[0][t] = 0
                
    # make sure we have a start tag at the beginning
    # reference: https://www.w3schools.com/python/python_dictionaries_access.asp
    keysss = TAG[0].keys()
    for t in keysss:
        last_TAG[0][t] = '<s>'
    
    # continue find cumulartive probabilities
    for i in range(1, len(stnce)):
        if stnce[i] not in frequent:
            stnce[i] = '<unk>'
            
    for i in range(1, len(stnce)):
        a = TAG[i-1].keys()
        for t in a:
            for wt in tag_list:
                if '('+wt + '~' + t+')' in transition:
                    if wt in TAG[i]:
                        try:
                            # some index may not exist so need to use try statement
                            b = transition['('+wt + '~' + t+')']* TAG[i-1][t] * emission['('+stnce[i] + '~' + wt+')']
                            if  b > TAG[i][wt]:
                                TAG[i][wt] = b
                                last_TAG[i][wt] = t
                        except:
                            pass
                    else:
                        # some index may not exist so need to use try statement
                        try:
                            TAG[i][wt] = transition['('+wt + '~' + t+')']* TAG[i-1][t] * emission['('+stnce[i] + '~' + wt+')']
                            last_TAG[i][wt] = t
                        except:
                            TAG[i][wt] = 0

    # backward propogation
    TAG_pred = []
    TAG_val_list = list(TAG[len(stnce)-1].values())
    TAG_key_list = list(TAG[len(stnce)-1].keys())
    # the highest probability is at the last word, loop backward for previous words
    max_prob = max(TAG[len(stnce)-1].values()) # find the maximum
    max_index = TAG_val_list.index(max_prob) # the index of the maximum
    max_tag = TAG_key_list[max_index]
    TAG_pred.append(max_tag)
    
    # iterate through pre_pos
    for i in range(len(stnce)-1, 0, -1):
        try:
            max_tag = last_TAG[i][max_tag]
            TAG_pred.append(max_tag)
        except:
            # some index/pairs does not exist
            TAG_pred.append('unk')
        
    # TAG_pred is in reverse
    TAG_pred_good = []
    for i in range(len(TAG_pred)-1,-1,-1):
        TAG_pred_good.append(TAG_pred[i])
    return TAG_pred_good



# use viterbi to predict pos for dev
viterbi_pred_dev = []
for w in dev_word_true:
    viterbi_pred_dev.append(viterbi(w))
viterbi_pred_dev= sum(viterbi_pred_dev, [])
accuracy_v = accuracy_score(tag_acc, viterbi_pred_dev)
print("The accuracy score for viterbi decoding HMM is:",accuracy_v)

tag_pred_test_viterbi=[]
for i in test_words_true:
    tag_pred_test_viterbi.append(viterbi(i))
#tag_pred_test_viterbi

vtl =[]
for t in tag_pred_test_viterbi:
    q= 1
    for i in t:
        vtl.append((i,q))
        q+=1
vtll = pd.DataFrame(vtl,columns = ["tag","index3"])
viterbi_df = pd.concat([wll,vtll],axis=1)
viterbi_df = viterbi_df.drop('index3', axis=1)
first_column2 = viterbi_df.pop('index')
viterbi_df.insert(0, 'index', first_column2)
#viterbi_df

viterbi_df.to_csv("viterbi.csv",index=False)
np.savetxt('/Users/William/Downloads/natural-language-processing/Hidden_Markov_Model/data/viterbi.out', greedy_df, delimiter=r'\t ', header=r'\t '.join(viterbi_df.columns.values), fmt='%s', comments='', encoding=None)
print("success")