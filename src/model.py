from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing.sequence import skipgrams
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Reshape, Embedding, Dot, Input, LSTM
from tensorflow.keras.models import Sequential, Model
import pandas as pd


def yield_strings(file_path='skipgram/corpus.txt'):
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(yield_strings('skipgram/corpus.txt'))

word2idx = tokenizer.word_index
idx2word = {v: k for k, v in word2idx.items()}

vocab_size = len(word2idx) + 1
embedding_dim = 100

word_ids = [[word2idx[w] for w in text.text_to_word_sequence(s)] for s in yield_strings('skipgram/corpus.txt')]

skipgram_model = load_model('skipgram/models/skipgram_model.h5')
word_embedding_layer = skipgram_model.get_layer(index=2)
word_embedding_weights = word_embedding_layer.get_weights()[0][1:]

pr_1_input = Input(shape=(None,1,100))
pr_2_input = Input(shape=(None,1,100))
reshape_1 = Reshape((100,))(pr_1_input)
reshape_2 = Reshape((100,))(pr_2_input)

#Model will take 2 sentences and using a RNN will perform binary classification
#If the sentences are similar, the output will be 1, else 0
lstm_layer_1 = LSTM(units=100, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, input_shape=(1000, 100))(pr_1_input)
lstm_layer_2 = LSTM(units=100, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, input_shape=(1000, 100))(pr_2_input)
dense_layer = Dense(units=1, activation='sigmoid')([lstm_layer_1, lstm_layer_2])

model = Model(inputs=[pr_1_input, pr_2_input], outputs=dense_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())