import re
from flask import Flask, render_template, request, redirect, url_for
from pull_request_getter import get_pull_request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import text
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

model = load_model('skipgram/models/skipgram_model.h5')

word_embedding_layer = model.get_layer(index=2)
word_embedding_weights = word_embedding_layer.get_weights()[0][1:]

def yield_strings(file_path='corpus.txt'):
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(yield_strings('corpus.txt'))

word2idx = tokenizer.word_index


def compare_prs(pr_1, pr_2, algorithm):
    pr_1_data = get_pull_request(pr_1)
    pr_1_body = parse_body(pr_1_data.body)
    pr_2_data = get_pull_request(pr_2)
    pr_2_body = parse_body(pr_2_data.body)

    if algorithm == 'average':
        return average_prs(pr_1_body, pr_2_data)
    elif algorithm == 'wordmover':
        return wordmover(pr_1_body, pr_2_body)
    # elif algorithm == 'nn':
    #     return nn_algo(pr_1_body, pr_2_body)
    else:
        return 'Invalid algorithm'

def parse_body(body):
    body = re.sub(r'https?://[^\s]+', 'LINK', body)
    body = re.sub(r'!\[.*\]\(.*\)', 'IMAGE', body)
    body = re.sub(r'```.*```', 'CODE', body)
    return body

def average_prs(pr_1_body, pr_2_body):
    pr_1_body = tokenizer.texts_to_sequences([pr_1_body])[0]
    pr_2_body = tokenizer.texts_to_sequences([pr_2_body])[0]
    pr_1_body_embedding = np.mean(word_embedding_weights[pr_1_body], axis=0)
    pr_2_body_embedding = np.mean(word_embedding_weights[pr_2_body], axis=0)
    return np.linalg.norm(pr_1_body_embedding - pr_2_body_embedding)

def wordmover(pr_1_body, pr_2_body):
    pr_1_body = tokenizer.texts_to_sequences([pr_1_body])[0]
    pr_2_body = tokenizer.texts_to_sequences([pr_2_body])[0]
    smaller_body = pr_1_body if len(pr_1_body) < len(pr_2_body) else pr_2_body
    larger_body = pr_1_body if len(pr_1_body) > len(pr_2_body) else pr_2_body
    move_cost = 0
    for word in smaller_body:
        move_cost += calculate_move_cost(word, larger_body)
    return move_cost



def calculate_move_cost(word_1, larger_body):
    word_1_embedding = word_embedding_weights[word_1]
    min_cost = float('inf')
    for word_2 in larger_body:
        word_2_embedding = word_embedding_weights[word_2]
        cost = np.linalg.norm(word_1_embedding - word_2_embedding)
        if cost < min_cost:
            min_cost = cost
    return min_cost


app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def compare_prs():
    if request.method == 'POST':
        pr_1 = request.form['pr_1']
        pr_2 = request.form['pr_2']
        algorithm = request.form['algorithm']
        return redirect(url_for('compare_pr_route', pr_1=pr_1, pr_2=pr_2, algorithm=algorithm))

@app.route('/compare_prs/<pr_1>/<pr_2>/<algorithm>')
def compare_pr_route(pr_1, pr_2, algorithm):
    result = compare_prs(pr_1, pr_2, algorithm)
    return render_template('compare_prs.html', result=result)