import pandas as pd


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


if __name__ == '__main__':
    CSVCleaner.import_csvs(7)
