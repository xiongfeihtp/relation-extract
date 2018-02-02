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
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 40

trainpath = "./dataset/train.txt"
testpath = "./dataset/test_people+hasqita.txt"

print('RNN/Embed/Sent/Query={},{},{},{}'.format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE, QUERY_HIDDEN_SIZE))


def tokenize(sent):
    return [x.strip() for x in sent]


def load_dataset(filepath):
    data = []
    with open(filepath, "rb") as f:
        lines = f.readlines()
        for line in lines:
            line = line.decode('utf-8').strip()
            com = line.split('\t')
            story = tokenize(com[3])
            question = tokenize(com[1] + '和' + com[2])
            answer = com[0]
            data.append([story, question, answer])
    return data


# 向量化数据
def vectorize_data(data, word_idx, story_maxlen, question_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, question, answer in data:
        xs_inx = []
        xqs_inx = []
        for w in story:
            if w in word_idx:
                xs_inx.append(word_idx[w])
            else:
                xs_inx.append(word_idx['_PAD'])
        xs.append(xs_inx)

        for w in question:
            if w in word_idx:
                xqs_inx.append(word_idx[w])
            else:
                xqs_inx.append(word_idx['_PAD'])
        xqs.append(xqs_inx)
        y = np.zeros(LABEL_NUM)
        answer = int(answer)
        y[answer] = 1
        ys.append(y)
    return pad_sequences(xs, maxlen=story_maxlen), pad_sequences(xqs, maxlen=question_maxlen), np.array(ys)


train = load_dataset(trainpath)
test = load_dataset(testpath)

vocab = set()
for story, question, _ in train + test:
    vocab |= set(story + question)
vocab = sorted(vocab)
vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
word_idx['_PAD'] = 0

story_maxlen = max(map(len, (x for x, _, _ in train + test)))
question_maxlen = max(map(len, (x for _, x, _ in train + test)))

x, xq, y = vectorize_data(train, word_idx, story_maxlen, question_maxlen)
tx, txq, ty = vectorize_data(test, word_idx, story_maxlen, question_maxlen)

print('vocab={}'.format(vocab))
print('x.shape={}'.format(x.shape))
print('xq.shape={}'.format(xq.shape))
print('y.shape={}'.format(y.shape))
print('story_maxlen,question_maxlen={},{}'.format(story_maxlen, question_maxlen))

print('Build model')

sentence = layers.Input(shape=(story_maxlen,), dtype='int32')
encoded_sentence = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
encoded_sentence = layers.Dropout(0.3)(encoded_sentence)

question = layers.Input(shape=(question_maxlen,), dtype='int32')
encoded_question = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
encoded_question = layers.Dropout(0.3)(encoded_question)
encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
#encoded_question输出的是RNN最后一个隐层输出，进行重复后和encoded_sentence按元素相加
encoded_question = layers.RepeatVector(story_maxlen)(encoded_question)

#直接元素相加
merged = layers.add([encoded_sentence, encoded_question])
merged = RNN(EMBED_HIDDEN_SIZE)(merged)
merged = layers.Dropout(0.3)(merged)

preds = layers.Dense(LABEL_NUM, activation='softmax')(merged)

# 输入和输出
model = Model([sentence, question], preds)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training")

model.fit([x, xq], y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.05)

loss, acc = model.evaluate([tx, txq], ty, batch_size=BATCH_SIZE)
print('test loss/test accuracy={:.4f}/{:.4f}'.format(loss, acc))
model.save("model/model1.h5")



