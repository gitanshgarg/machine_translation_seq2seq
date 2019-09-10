# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 23:18:07 2019

@author: GITANSH
"""

import os, sys

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


#download pre-trained glove word-embedding from # Direct link: http://nlp.stanford.edu/data/glove.6B.zip
#place them in a folder named glove in your working directory




# some config
BATCH_SIZE = 64  # Batch size for training.
EPOCHS = 100  # Number of epochs to train for.
LATENT_DIM = 256  # Latent dimensionality of the encoding space.
NUM_SAMPLES = 10000  # Number of samples to train on.
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

# Where we will store the data
input_texts = [] # sentence in original language
target_texts = [] # sentence in target language
target_texts_inputs = [] # sentence in target language offset by 1


# load in the data
# download the data at: http://www.manythings.org/anki/
t = 0
for line in open('ita.txt'):
  # only keep a limited number of samples
  t += 1
  if t > NUM_SAMPLES:
    break

  # input and target are separated by tab
  if '\t' not in line:
    continue

  # split up the input and translation
  input_text,translation = line.rstrip().split('\t')

  # make the target input and output
  # recall we'll be using teacher forcing
  target_text = translation + ' <eos>'
  target_text_input = '<sos> ' + translation

  input_texts.append(input_text)
  target_texts.append(target_text)
  target_texts_inputs.append(target_text_input)
print("num samples:", len(input_texts))


# tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

# get the word to index mapping for input language
word2idx_inputs = tokenizer_inputs.word_index
print('Found %s unique input tokens.' % len(word2idx_inputs))

# determine maximum length input sequence
max_len_input = max(len(s) for s in input_sequences)

# tokenize the outputs
# don't filter out special characters
# otherwise <sos> and <eos> won't appear
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs) # inefficient, oh well
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

# get the word to index mapping for output language
word2idx_outputs = tokenizer_outputs.word_index
print('Found %s unique output tokens.' % len(word2idx_outputs))

# store number of output words for later
# remember to add 1 since indexing starts at 1
num_words_output = len(word2idx_outputs) + 1

# determine maximum length output sequence
max_len_target = max(len(s) for s in target_sequences)


# pad the sequences
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
print("encoder_inputs.shape:", encoder_inputs.shape)
print("encoder_inputs[0]:", encoder_inputs[0])

decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
print("decoder_inputs[0]:", decoder_inputs[0])
print("decoder_inputs.shape:", decoder_inputs.shape)

decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')







# store all the pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('glove/glove.6B.%sd.txt' % EMBEDDING_DIM),encoding="utf-8") as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))




# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx_inputs.items():
  if i < MAX_NUM_WORDS:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector




# create embedding layer
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=max_len_input,
  # trainable=True
)


# create targets, since we cannot use sparse
# categorical cross entropy when we have sequences
decoder_targets_one_hot = np.zeros(
  (
    len(input_texts),
    max_len_target,
    num_words_output
  ),
  dtype='float32'
)

# assign the values
for i, d in enumerate(decoder_targets):
  for t, word in enumerate(d):
      if(word>0):
          decoder_targets_one_hot[i, t, word] = 1





#build the model
encoder_inp=Input(shape=(max_len_input,));
lstm=LSTM(LATENT_DIM,return_state=True);
x=embedding_layer(encoder_inp);
x,h,c=lstm(x);

encoder_states=[h,c];

decoder_inp=Input(shape=(max_len_target,))




decoder_embedding=Embedding(num_words_output,LATENT_DIM);
decoder_lstm=LSTM(LATENT_DIM,return_state=True,return_sequences=True);
x=decoder_embedding(decoder_inp);
x,_,_=decoder_lstm(x,initial_state=encoder_states);
decoder_dense=Dense(num_words_output,activation="softmax")

decoder_output=decoder_dense(x);


model_train=Model([encoder_inp,decoder_inp],decoder_output);

# Compile the model and train it
model_train.compile(
  optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)
r = model_train.fit(
  [encoder_inputs, decoder_inputs], decoder_targets_one_hot,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=0.2,
)

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

# Save model
model_train.save('s2s.h5')


#building the sampling model
encoder_model=Model(encoder_inp,encoder_states);


decoder_inp_s=Input(shape=(1,));
decoder_h=Input(shape=(LATENT_DIM,));
decoder_c=Input(shape=(LATENT_DIM,));
decoder_states=[decoder_h,decoder_c];

x1=decoder_embedding(decoder_inp_s);
x1,h,c=decoder_lstm(x1,initial_state=decoder_states)
decoder_out=[h,c];
decoder_out_s=decoder_dense(x1);


decoder_model=Model([decoder_inp_s]+decoder_states,[decoder_out_s]+decoder_out);
idx2word_out={k:v for v,k in word2idx_outputs.items() };

def translation2(input_text):
    encoder_states=encoder_model.predict(input_text);
    
    
    inp=np.zeros((1,1));
    inp[0][0]=word2idx_outputs['<sos>'];
    output_sen=[];
    eos=word2idx_outputs['<eos>'];
    for i in range(max_len_target):
        o,h,c=decoder_model.predict([inp]+encoder_states);
        prob=o[0,0];
        prob[0]=0;
        prob/=prob.sum();
        
        idx=np.argmax(prob);
        
        if(idx==eos):
            break;
            
        output_sen.append(idx2word_out.get(idx));
        encoder_states=[h,c];
        inp[0,0]=idx;

    return " ".join(output_sen);




while True:
  # Do some test translations
  i = np.random.choice(len(input_texts))
  input_seq = encoder_inputs[i:i+1]
  translation = translation2(input_seq)
  print('-')
  print('Input:', input_texts[i])
  print('Translation:', translation)

  ans = input("Continue? [Y/n]")
  if ans and ans.lower().startswith('n'):
    break
    
    


        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    


























          
          

          
          
          









