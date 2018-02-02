# -*- coding: utf-8 -*-
from __future__ import print_function
from functools import reduce
import re
import tarfile

import numpy as np
import sys
from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent,merge,TimeDistributed
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, RepeatVector,Permute,Reshape
from keras.regularizers import l2
from keras import backend as K

def get_R(X):
    Y, alpha = X[0], X[1]
    ans = K.T.batched_dot(Y, alpha)
    return ans

LABEL_NUM = 12
RNN = recurrent.LSTM
CHAR_EMBED_HIDDEN_SIZE = 50
WORD_EMBED_HIDDEN_SIZE = 100
RNN_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 40
BI_WORD_EMBED = 200
trainpath = "./dataset/train_tokenize.txt"
testpath = "./dataset/test_tokenize.txt"
pre_train_embedding = "./wordvec/segdata1.txt.vec2"
print('RNN / Char_Embed / Word_Embed / Hidden_Dim = {}, {}, {}, {}'.format(RNN,
CHAR_EMBED_HIDDEN_SIZE,
WORD_EMBED_HIDDEN_SIZE,
RNN_HIDDEN_SIZE))

def tokenize(sent):
    return [x.strip() for x in sent]

def load_embedding(word_idx,filepath,embedding_size):
    embeddings = np.random.normal(0.00,1.00,[len(word_idx),embedding_size])
    count = 0
    with open(filepath,"rb") as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.decode('utf-8').strip()
            word = line.rstrip().split(' ')[0]
            if(word in word_idx):
                count += 1
                vec = line.strip().split(' ')[1:]
                vec = np.array(vec)
                embeddings[word_idx[word]] = vec
    embeddings[word_idx['_PAD']] = np.zeros(embedding_size)
    print('the match count:',count)
    return embeddings

def load_dataset(filepath):
    data = []
    with open(filepath,"rb") as f:
        lines = f.readlines()
        for line in lines:
            line = line.decode('utf-8').strip()
            com = line.split('\t')
            story = tokenize(com[3])
            story_token = com[4].split('/')
            question = tokenize(com[1]+'和'+com[2])
            question_token = [com[1],com[2]]
            answer = com[0]
            data.append([story,story_token,question,question_token,answer])
    return data

def vectorize_data(data,char_idx,word_idx,storyc_maxlen,storyw_maxlen,questionc_maxlen,questionw_maxlen):
    xs = []
    xts = []
    xqs = []
    xqts = []
    ys = []
    for storyc,storyw,questionc,questionw,answer in data:
        xs_inx = []
        xts_inx = []
        xqs_inx = []
        xqts_inx = []
        for w in storyc:
            if w in char_idx:
                xs_inx.append(char_idx[w])
            else:
                xs_inx.append(char_idx['_PAD'])
        xs.append(xs_inx)
        for w in storyw:
            if w in word_idx:
                xts_inx.append(word_idx[w])
            else:
                xts_inx.append(word_idx['_PAD'])
        xts.append(xts_inx)
        for w in questionc:
            if w in char_idx:
                xqs_inx.append(char_idx[w])
            else:
                xqs_inx.append(char_idx['_PAD'])
        xqs.append(xqs_inx)
        for w in questionw:
            if w in word_idx:
                xqts_inx.append(word_idx[w])
            else:
                xqts_inx.append(word_idx['_PAD'])
        xqts.append(xqts_inx)
        y = np.zeros(LABEL_NUM)
        answer = int(answer)
        y[answer] = 1
        ys.append(y)
    return pad_sequences(xs,maxlen=storyc_maxlen),pad_sequences(xts,maxlen=storyw_maxlen),pad_sequences(xqs,maxlen=questionc_maxlen),pad_sequences(xqts,maxlen=questionw_maxlen),np.array(ys)

train=load_dataset(trainpath)
test=load_dataset(testpath)

char_vocab = set()
word_vocab = set()
for storyc,storyw,questionc,questionw,_ in train+test:
    char_vocab |= set(storyc+questionc)
    word_vocab |= set(storyw+questionw)
char_vocab = sorted(char_vocab)
char_vocab_size = len(char_vocab)+1
word_vocab = sorted(word_vocab)
word_vocab_size = len(word_vocab)+1
char_idx = dict((c,i+1) for i,c in enumerate(char_vocab))
char_idx['_PAD'] = 0
word_idx = dict((c,i+1) for i,c in enumerate(word_vocab))
word_idx['_PAD'] = 0
storyc_maxlen = max(map(len,(x for x,_,_,_,_ in train+test)))
questionc_maxlen = max(map(len,(x for _,_,x,_,_ in train+test)))
storyw_maxlen = max(map(len,(x for _,x,_,_,_ in train+test)))
questionw_maxlen = max(map(len,(x for _,_,_,x,_ in train+test)))

x,xt,xq,xqt,y = vectorize_data(train,char_idx,word_idx,storyc_maxlen,storyw_maxlen,questionc_maxlen,questionw_maxlen)
tx,txt,txq,txqt,ty = vectorize_data(test,char_idx,word_idx,storyc_maxlen,storyw_maxlen,questionc_maxlen,questionw_maxlen)
init_embeddings = load_embedding(word_idx,pre_train_embedding,WORD_EMBED_HIDDEN_SIZE)

embedding_layer = layers.Embedding(word_vocab_size,
                                   WORD_EMBED_HIDDEN_SIZE,
                                   weights=[init_embeddings],
                                   trainable=False)

print('char_vocab={}'.format(char_vocab))
print('x.shape={}'.format(x.shape))
print('xt.shape={}'.format(xt.shape))
print('xq.shape={}'.format(xq.shape))
print('xqt.shape={}'.format(xqt.shape))
print('y.shape={}'.format(y.shape))
print('storyc_maxlen,questionc_maxlen,storyw_maxlen,questionw_maxlen={},{}'.format(storyc_maxlen,questionc_maxlen,storyw_maxlen,questionw_maxlen))
print('Build model...')

#sentence_c = layers.Input(shape=(storyc_maxlen,),dtype='int32')
sentence_w = layers.Input(shape=(storyw_maxlen,),dtype='int32')
#encoded_sentence_char = layers.Embedding(char_vocab_size,CHAR_EMBED_HIDDEN_SIZE)(sentence_c)
#encoded_sentence_char = layers.Dropout(0.3)(encoded_sentence_char)
encoded_sentence_word = embedding_layer(sentence_w)
encoded_sentence_word = layers.Dropout(0.3)(encoded_sentence_word)

#question_c = layers.Input(shape=(questionc_maxlen,),dtype='int32')
question_w = layers.Input(shape=(questionw_maxlen,),dtype='int32')
#encoded_question_char = layers.Embedding(char_vocab_size,CHAR_EMBED_HIDDEN_SIZE)(question_c)
#encoded_question_char = layers.Dropout(0.3)(encoded_question_char)
encoded_question_word = embedding_layer(question_w)
encoded_question_word = Dropout(0.3)(encoded_question_word)

encoded_sentence_word_forward = RNN(RNN_HIDDEN_SIZE)(encoded_sentence_word)
encoded_sentence_word_backward = RNN(RNN_HIDDEN_SIZE,go_backwards=True)(encoded_sentence_word)
encoded_sentence_word = layers.concatenate([encoded_sentence_word_forward,encoded_sentence_word_backward],axis=-1)

encoded_question_word_forward = RNN(RNN_HIDDEN_SIZE)(encoded_question_word)
encoded_question_word_backward = RNN(RNN_HIDDEN_SIZE)(encoded_question_word)
encoded_question_word = layers.concatenate([encoded_question_word_forward,encoded_question_word_backward],axis=-1)

#encoded_question_char_forward = RNN(RNN_HIDDEN_SIZE)(encoded_question_char)
#encoded_question_char_backward = RNN(RNN_HIDDEN_SIZE,go_backwards=True)(encoded_question_char)
#encoded_question_char = layers.concatenate([encoded_question_char_forward,encoded_question_char_backward],axis=-1)
#encoded_question_char = layers.RepeatVector(storyc_maxlen)(encoded_question_char)

#merged = layers.concatenate([encoded_sentence_char,encoded_question_char],axis=-1)
#merged = RNN(RNN_HIDDEN_SIZE)(merged)

#加入attention层
# hop 1
w_aspect = encoded_question_word
w_aspects = RepeatVector(storyw_maxlen, name="w_aspects1")(w_aspect)
w_context = RepeatVector(storyw_maxlen, name="w_context")(encoded_sentence_word)
merged = merge([w_context, w_aspects], name='merged1', mode='concat')
distributed = TimeDistributed(layers.Dense(1, W_regularizer=l2(0.01), activation='tanh'), name="distributed1")(merged)
flat_alpha = Flatten(name="flat_alpha1")(distributed)
alpha = Dense(storyw_maxlen, activation='softmax', name="alpha1")(flat_alpha)
w_context_trans = Permute((2, 1), name="w_context_trans1")(w_context)
r_ = merge([w_context_trans, alpha], output_shape=(BI_WORD_EMBED, 1), name="r_1", mode=get_R)
r = Reshape((BI_WORD_EMBED,), name="r1")(r_)
w_aspect_linear = Dense(BI_WORD_EMBED, W_regularizer=l2(0.01), activation='linear')(w_aspect)
merged = merge([r, w_aspect_linear], mode='sum')
#w_aspect = Dense(emb, W_regularizer=l2(0.01), name="w_aspect_2")(merged)
#merged = layers.concatenate([merged,encoded_sentence_word,encoded_question_word],axis=-1)
#merged = layers.concatenate([encoded_sentence_word,encoded_question_word],axis=-1)
#merged = layers.Dropout(0.3)(merged)
preds = layers.Dense(LABEL_NUM, activation='softmax')(merged)

model = Model([sentence_w,question_w,], preds)
model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])

print('Training')
model.fit([xt,xqt], y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.05)
loss, acc = model.evaluate([txt,txqt], ty,
                           batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
model.save('model/model1_attention.h5')
#model = load_model('model/model1.h5')
#loss, acc = model.evaluate([tx, txq], ty,
# batch_size=BATCH_SIZE)
#print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))