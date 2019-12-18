from __future__ import print_function
import numpy as np
import pandas as pd
import torch as t
from sklearn.model_selection import train_test_split
from Lstmclassifier import Brain

#Hyperparameter
##########################
embedding_dim = 512 #embedding dimensions that needs to be given in the nn.embedding
hidden_dim = 512 #Hidden dimension to be given in the neural nets
# in this case both embedding dim and hidden is same due to single layer, in case of multiple layers multiple hidden dimension can be used.
output_size = 2 # size of the output. Since its a binary classification, the size is 2
epochs = 500 # No of iteration training needs to be done on the data set
##########################
#Load the file
##########################
filename = 'E:\AI_files\Sentiment_analysis\data_mod_filtered.csv' # path from where dataset is stored.
df = pd.read_csv(filename,encoding = "ISO-8859-1")
##########################
#Removing punctuations
##########################
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

for i in range(len(df.iloc[:,1])):
        no_punct = ""
        for char in df.iloc[i,1]:
            if char not in punctuations:
                no_punct = no_punct + char
        df.iloc[i,1] = no_punct
##########################

#tokenizing function
##########################
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq] #converts words into individual tokens
    return t.tensor(idxs, dtype=t.long)
##########################
#Preparing the dictionary
##########################
word_to_ix = {}
for i in range(len(df.iloc[:,1])):
    for word in (df.iloc[i,1].split()):
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix) #adds each words into the dictionary

vocab_size = len(word_to_ix)
##########################

#Defining the model
##########################
nn_Brain = Brain(embedding_dim,hidden_dim, vocab_size,output_size)
##########################
#splitting the dataset into test and training set
##########################
samp_train, samp_test, label_train, label_test = train_test_split(df.iloc[:,1], df.iloc[:,0], test_size= 0.2, random_state=49)
samp_train = samp_train.values
##########################
#Training the model
##########################
nn_Brain.pre_trainer(samp_train,label_train,word_to_ix,epochs)
##########################

#Testing the trained model
##########################
samp_test = samp_test.values
label_test = label_test.values
n=0
for i in range(len(samp_test)):
    sentence = prepare_sequence(samp_test[i].split(),word_to_ix)
    out_data = nn_Brain.classify(sentence)
    out_data = out_data.numpy()
    print(out_data)
    print(label_test[i])
    if out_data == label_test[i]:
        n+=1
accuracy = n / len(samp_test) *100
print(accuracy)
##########################

#print(label_train.shape)
#print(labels_class)






