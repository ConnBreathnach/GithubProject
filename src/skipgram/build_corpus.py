import pandas as pd

def generate_corpus(csv_path):
    df = pd.read_csv(csv_path)
    count = 0
    with open('corpus.txt', 'w') as f:
        for row in df.iterrows():
            f.write(row[1]['pr_title'] + '\n')
            f.write(row[1]['pr_body'] + '\n')
            count += 1
            if count % 1000 == 0:
                print(count/df.shape[0] * 100, '%')


if __name__ == '__main__':
    generate_corpus('../../data/final_dataset_2.csv')
    print('Done!')


