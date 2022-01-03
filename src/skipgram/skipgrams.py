from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing.sequence import skipgrams
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Reshape, Embedding, Dot, Input
from tensorflow.keras.models import Sequential, Model
import pandas as pd

def yield_strings(file_path='corpus.txt'):
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(yield_strings('corpus.txt'))

word2idx = tokenizer.word_index
idx2word = {v: k for k, v in word2idx.items()}

vocab_size = len(word2idx) + 1
embedding_dim = 100

word_ids = [[word2idx[w] for w in text.text_to_word_sequence(s)] for s in yield_strings('corpus.txt')]
print('Vocab size:', vocab_size)
print('Vocab sample:', list(word2idx.items())[:10])


# skip_grams = [skipgrams(word_id, vocabulary_size=vocab_size, window_size=10) for word_id in word_ids]
# pairs, labels = skip_grams[0][0], skip_grams[0][1]
# for i in range(10):
#     print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
#           idx2word[pairs[i][0]], pairs[i][0],
#           idx2word[pairs[i][1]], pairs[i][1],
#           labels[i]))
#
# word_input = Input(shape=(1,), dtype='int32')
# word_embedding = Embedding(vocab_size, embedding_dim,
#                          embeddings_initializer="glorot_uniform",
#                          input_length=1)(word_input)
# word_reshape = Reshape((embedding_dim,))(word_embedding)
#
# context_input = Input(shape=(1,), dtype='int32')
# context_embedding = Embedding(vocab_size, embedding_dim,
#                   embeddings_initializer="glorot_uniform",
#                   input_length=1)(context_input)
# context_reshape = Reshape((embedding_dim,))(context_embedding)
#
#
# dotted_layer = Dot(axes=1)([word_reshape, context_reshape])
# output_layer = Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid")(dotted_layer)
# skipgram_model = Model(inputs=[word_input, context_input], outputs=output_layer)
# skipgram_model.compile(loss="mean_squared_error", optimizer="rmsprop")
#
# print(skipgram_model.summary())
#
#
# for epoch in range(0, 5):
#     loss = 0
#     print("Epoch:", epoch)
#     for i, elem in enumerate(skip_grams):
#         if elem == ([], []):
#             continue
#         pair_first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
#         pair_second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
#         label_elem = np.array(elem[1], dtype='int32')
#         X = [pair_first_elem, pair_second_elem]
#         Y = label_elem
#         if i % 10000 == 0:
#             print('Processed {} (skip_first, skip_second, relevance) pairs'.format(i))
#             print('Loss:', loss)
#         loss += skipgram_model.train_on_batch(X, Y)
#     # merge_layer = skipgram_model.layers[0]
#     # word_embedding_weights = merge_layer.get_weights()[0]
#     # trained_word_model = merge_layer.layers[0]
#     # trained_word_model.save('models/word_model{}.h5'.format(epoch))
#     skipgram_model.save('models/skipgram_model{}.h5'.format(epoch))
#     print('Epoch:', epoch, 'Loss:', loss)
#
# # merge_layer = skipgram_model.layers[0]
# # word_embedding_weights = merge_layer.get_weights()[0]
# # trained_word_model = merge_layer.layers[0]
# # trained_word_model.save('models/word_model.h5')
# skipgram_model.save('models/skipgram_model.h5')
# print('Saved models')

model = load_model('models/skipgram_model.h5')
word_embedding_layer = model.get_layer(index=2)
word_embedding_weights = word_embedding_layer.get_weights()[0][1:]
print(pd.DataFrame(word_embedding_weights, index=idx2word.values()).head())
from sklearn.metrics.pairwise import euclidean_distances

distance_matrix = euclidean_distances(word_embedding_weights)
print(distance_matrix.shape)

similar_words = {search_term: [idx2word[idx] for idx in distance_matrix[word2idx[search_term]-1].argsort()[1:6]+1]
                   for search_term in ['code', 'link', 'fix', 'bug', 'error']}

print(similar_words)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

words = sum([[k] + v for k, v in similar_words.items()], [])
words_ids = [word2idx[w] for w in words]
word_vectors = np.array([word_embedding_weights[idx] for idx in words_ids])
print('Total words:', len(words), '\tWord Embedding shapes:', word_vectors.shape)

tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=3)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(word_vectors)
labels = words

plt.figure(figsize=(14, 8))
plt.scatter(T[:, 0], T[:, 1], c='steelblue', edgecolors='k')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')
