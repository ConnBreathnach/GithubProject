import pandas as pd
import re
from langdetect import detect

class CSVCleaner:
    # Read the files dataset{num}.csv and combine them into one dataframe
    def import_csvs(num):
        df = pd.read_csv('../data/dataset.csv')
        for i in range(1, num + 1):
            try:
                temp = pd.read_csv('../data/dataset{}.csv'.format(i))
                df = df.append(temp)
            except FileNotFoundError:
                print('No dataset{}.csv'.format(i))
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        df.drop_duplicates(inplace=True)
        df.to_csv('../data/final_dataset.csv', index=True)
        return df

    def normalize_data(self, file_name):
        df = pd.read_csv(file_name)
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        df.drop_duplicates(inplace=True)

    def remove_non_english(file_name):
        df = pd.read_csv(file_name)
        for index, row in df.iterrows():
            try:
                if detect(row['pr_body']) != 'en':
                    df.drop(index, inplace=True)
            except:
                df.drop(index, inplace=True)
        df.to_csv('../data/final_dataset_2.csv', index=True)







if __name__ == '__main__':
    CSVCleaner.remove_non_english('../data/final_dataset.csv')
