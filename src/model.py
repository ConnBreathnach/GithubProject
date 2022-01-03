from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing.sequence import skipgrams
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Reshape, Embedding, Dot, Input, LSTM
from tensorflow.keras.models import Sequential, Model
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import convert_to_tensor


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
word_embedding_weights = word_embedding_layer.get_weights()[0]

input_1 = Input(shape=(1,))
input_2 = Input(shape=(1,))

embedding_layer_1 = Embedding(vocab_size, embedding_dim, weights=[word_embedding_weights], input_length=1, trainable=False)(input_1)
embedding_layer_2 = Embedding(vocab_size, embedding_dim, weights=[word_embedding_weights], input_length=1, trainable=False)(input_2)
#Model will take 2 sentences and using a RNN will perform binary classification
#If the sentences are from same repo, the output will be 1, else 0
lstm_layer_1 = LSTM(units=100, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, input_shape=(1000, 100))(embedding_layer_1)
lstm_layer_2 = LSTM(units=100, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, input_shape=(1000, 100))(embedding_layer_2)
merge_layer = Dot(axes=1)([lstm_layer_1, lstm_layer_2])
dense_layer = Dense(units=1, activation='sigmoid')(merge_layer)

model = Model(inputs=[input_1, input_2], outputs=dense_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

dataset = pd.read_csv('../data/combined_dataset.csv')
dataset.drop(['Unnamed: 0', 'pr_title_1', 'pr_title_2', 'commits_1', 'commits_2', 'repo_1_name', 'repo_1_user', 'repo_2_name', 'repo_2_user'], axis=1, inplace=True)

X = dataset.drop(['same_repo'], axis=1)
y = dataset['same_repo']
X = X.apply(tokenizer.texts_to_sequences)
# X = X.applymap(np.array, dtype=object)
print(X.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train=np.asarray(X_train).astype(object)
X_test=np.asarray(X_test).astype(object)

model.fit([X_train['pr_body_1'], X_train['pr_body_2']], y_train, epochs=5, batch_size=32, validation_data=([X_test['pr_body_1'], X_test['pr_body_2']], y_test))
model.save('same_rep_model.h5')