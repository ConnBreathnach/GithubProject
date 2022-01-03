from tensorflow.keras.models import load_model

model = load_model('models/skipgram_model.h5')
word_embed_layer = model.get_layer('embedding')
weights = word_embed_layer.get_weights()[0][1:]

from sklearn.metrics.pairwise import euclidean_distances

distance_matrix = euclidean_distances(weights)
print(distance_matrix.shape)

similar_words = {search_term: [id2word[idx] for idx in distance_matrix[word2id[search_term]-1].argsort()[1:6]+1]
                   for search_term in ['god', 'jesus', 'noah', 'egypt', 'john', 'gospel', 'moses','famine']}

similar_words