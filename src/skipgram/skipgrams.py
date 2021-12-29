from keras.preprocessing import text

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