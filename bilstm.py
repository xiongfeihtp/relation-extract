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
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

LABEL_NUM = 12
RNN = recurrent.LSTM

CHAR_EMBED_HIDDEN_SIZE = 50
WORD_EMBED_HIDDEN_SIZE = 100


RNN_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 40
trainpath = "./dataset/train_tokenize.txt"
testpath = "./dataset/test_tokenize.txt"
pre_train_embedding = "./wordvec/segdata1.txt.vec2"
print('RNN / Char_Embed / Word_Embed / Hidden_Dim = {}, {}, {}, {}'.format(RNN,
                                                                           CHAR_EMBED_HIDDEN_SIZE,
                                                                           WORD_EMBED_HIDDEN_SIZE,
                                                                           RNN_HIDDEN_SIZE))
def tokenize(sent):
    return [x.strip() for x in sent]


def load_embedding(word_idx, filepath, embedding_size):
    embeddings = np.random.normal(0.00, 1.00, [len(word_idx), embedding_size])
    count = 0
    #读取文件时，根据特定的格式进行读取
    with open(filepath, "rb") as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.decode('utf-8').strip()
            word = line.rstrip().split(' ')[0]
            if (word in word_idx):
                count += 1
                vec = line.strip().split(' ')[1:]
                vec = np.array(vec)
                embeddings[word_idx[word]] = vec
    #_PAD映射为零
    embeddings[word_idx['_PAD']] = np.zeros(embedding_size)
    print('the match count:', count)
    #匹配出现过的词，其他在词向量表中未登录词取随机
    return embeddings

def load_dataset(filepath):
    data = []
    with open(filepath, "rb") as f:
        lines = f.readlines()
    for line in lines:
        line = line.decode('utf-8').strip()
        com = line.split('\t')
        story = tokenize(com[3])
        story_token = com[4].split('/')
        question = tokenize(com[1] + '和' + com[2])
        question_token = [com[1], com[2]]
        answer = com[0]
        data.append([story, story_token, question, question_token, answer])
    return data


def vectorize_data(data, char_idx, word_idx, storyc_maxlen, storyw_maxlen, questionc_maxlen, questionw_maxlen):
    xs = []
    xts = []
    xqs = []
    xqts = []
    ys = []
    for storyc, storyw, questionc, questionw, answer in data:
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
        #对输入进行对齐和填零，对将输出列表转换为array
    return pad_sequences(xs, maxlen=storyc_maxlen), \
           pad_sequences(xts, maxlen=storyw_maxlen), \
           pad_sequences(xqs,maxlen=questionc_maxlen), \
           pad_sequences(xqts, maxlen=questionw_maxlen), np.array(ys)

train = load_dataset(trainpath)
test = load_dataset(testpath)


#char vector and word vector
char_vocab = set()
word_vocab = set()
for storyc, storyw, questionc, questionw, _ in train + test:
    char_vocab |= set(storyc + questionc)
    word_vocab |= set(storyw + questionw)
char_vocab = sorted(char_vocab)
char_vocab_size = len(char_vocab) + 1
word_vocab = sorted(word_vocab)
word_vocab_size = len(word_vocab) + 1

#创建字典索引
char_idx = dict((c, i + 1) for i, c in enumerate(char_vocab))
char_idx['_PAD'] = 0
word_idx = dict((c, i + 1) for i, c in enumerate(word_vocab))
word_idx['_PAD'] = 0
#统计最大长度
storyc_maxlen = max(map(len, (x for x, _, _, _, _ in train + test)))
questionc_maxlen = max(map(len, (x for _, _, x, _, _ in train + test)))
storyw_maxlen = max(map(len, (x for _, x, _, _, _ in train + test)))
questionw_maxlen = max(map(len, (x for _, _, _, x, _ in train + test)))

x, xt, xq, xqt, y = vectorize_data(train, char_idx, word_idx, storyc_maxlen, storyw_maxlen, questionc_maxlen,
                                   questionw_maxlen)
tx, txt, txq, txqt, ty = vectorize_data(test, char_idx, word_idx, storyc_maxlen, storyw_maxlen, questionc_maxlen,
                                        questionw_maxlen)

#导入预训练词向量
init_embeddings = load_embedding(word_idx, pre_train_embedding, WORD_EMBED_HIDDEN_SIZE)

#word_vocab_size=最大下标+1，trainable=False，不进行微调，因为数据量比较小
embedding_layer = layers.Embedding(word_vocab_size, WORD_EMBED_HIDDEN_SIZE, weights=[init_embeddings], trainable=False)

print('char_vocab={}'.format(char_vocab))
print('x.shape={}'.format(x.shape))
print('xt.shape={}'.format(xt.shape))
print('xq.shape={}'.format(xq.shape))
print('xqt.shape={}'.format(xqt.shape))
print('y.shape={}'.format(y.shape))
print('storyc_maxlen,questionc_maxlen,storyw_maxlen,questionw_maxlen={},{}'.format(storyc_maxlen, questionc_maxlen,
                                                                                   storyw_maxlen, questionw_maxlen))
print('Build model...')

sentence_c = layers.Input(shape=(storyc_maxlen,), dtype='int32')
sentence_w = layers.Input(shape=(storyw_maxlen,), dtype='int32')
#可调embedding layer
encoded_sentence_char = layers.Embedding(char_vocab_size, CHAR_EMBED_HIDDEN_SIZE)(sentence_c)
encoded_sentence_char = layers.Dropout(0.3)(encoded_sentence_char)
#预训练embedding layer不可训练
encoded_sentence_word = embedding_layer(sentence_w)
encoded_sentence_word = layers.Dropout(0.3)(encoded_sentence_word)

question_c = layers.Input(shape=(questionc_maxlen,), dtype='int32')
question_w = layers.Input(shape=(questionw_maxlen,), dtype='int32')
encoded_question_char = layers.Embedding(char_vocab_size,
                                         CHAR_EMBED_HIDDEN_SIZE)(question_c)
encoded_question_char = layers.Dropout(0.3)(encoded_question_char)
encoded_question_word = embedding_layer(question_w)
encoded_question_word = layers.Dropout(0.3)(encoded_question_word)

#双向LSTM单元
encoded_sentence_word_forward = RNN(RNN_HIDDEN_SIZE)(encoded_sentence_word)
encoded_sentence_word_backward = RNN(RNN_HIDDEN_SIZE, go_backwards=True)(encoded_sentence_word)
encoded_sentence_word = layers.concatenate([encoded_sentence_word_forward, encoded_sentence_word_backward], axis=-1)
#双向LSTM单元
encoded_question_word_forward = RNN(RNN_HIDDEN_SIZE)(encoded_question_word)
encoded_question_word_backward = RNN(RNN_HIDDEN_SIZE)(encoded_question_word)
encoded_question_word = layers.concatenate([encoded_question_word_forward, encoded_question_word_backward], axis=-1)
#双向LSTM单元
encoded_question_char_forward = RNN(RNN_HIDDEN_SIZE)(encoded_question_char)
encoded_question_char_backward = RNN(RNN_HIDDEN_SIZE, go_backwards=True)(encoded_question_char)
encoded_question_char = layers.concatenate([encoded_question_char_forward, encoded_question_char_backward], axis=-1)

#一个embedding层和一个RNN输出进行操作时，会出现这种情况
encoded_question_char = layers.RepeatVector(storyc_maxlen)(encoded_question_char)
merged = layers.concatenate([encoded_sentence_char, encoded_question_char], axis=-1)
#单向LSTM
merged = RNN(RNN_HIDDEN_SIZE)(merged)
merged = layers.concatenate([merged, encoded_sentence_word, encoded_question_word], axis=-1)
merged = layers.Dropout(0.3)(merged)
#非输出层的全连接要加dropout层
preds = layers.Dense(LABEL_NUM, activation='softmax')(merged)

model = Model([sentence_c, sentence_w, question_c, question_w, ], preds)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('Training')
model.fit([x, xt, xq, xqt], y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.05)
loss, acc = model.evaluate([tx, txt, txq, txqt], ty, batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
model.save('model/model1.h5')
# model = load_model('model/model1.h5')
# loss, acc = model.evaluate([tx, txq], ty,
# batch_size=BATCH_SIZE)
# print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
