#IMPORTING INSALLED ENV.

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn 
import tensorflow as tf
import random
import json
import pickle

#IMPORTING JSON. (CONTAINING TRAINING SET)

with open('intents.json') as file:
    data= json.load(file)


#LOOPING THROUGH THE JSON DATA

try:
    with open('data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f)
    
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []


    #STEMMING AND TOKENIZATION

    for intent in data['intents']:
        for pattern in intent ['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])
        
        if intent ['tag'] not in labels:
            labels.append(intent['tag'])



    #STEMMING AND SORTING SET DUPLICATE IN VOCABULARY WORDS IN DATA

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    #BAG OF WORDS INPUTS

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
            bag = []

            wrds = [stemmer.stem(w.lower()) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)
                    
        #GENERATED OUTPUT FOR THE BAG OF WORDS

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1


        #TRAINING LIST AND OUTPUT LISTS

            training.append(bag)
            output.append(output_row)

    
    #ARRAY OF THE TRAINING DATA

    training = np.array(training)
    output = np.array(output)



    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)

#BUILDING THE MODEL

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = 'softmax')
net = tflearn.regression(net)


model = tflearn.DNN(net)

#PASSING TRAINING DATA & NOT STRESSING TO ALWAYS RETRAIN MODELS (NB: epoch - 'no. of times the machine sees the data to make better predictions'

try:
    model.load('model.tflearn')
except:
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save('model.tflearn')


#MAKING PREDICTIONS 

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return np.array(bag)


#USER INPUT AND BOT RESPONSE 

def chat():
    print('Drop me message make i get back to you(type quit if you don finish)!')
    while True:
        inp = input('You: ')
        if inp.lower() == 'quit':
            break

        #Model Response Probability
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]


        if results[results_index] > 0.7:
          for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
                         
          print(random.choice(responses)) 
        else:
            print('No vex i been no get wetin you type. Type another thing abeg')


chat()



