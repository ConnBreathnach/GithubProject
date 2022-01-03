import pandas as pd
from pull_request_getter import PullRequestGetter
import re
import random

class DatasetGenerator:
    def __init__(self, data_path='../data/', size=1000000):
        self.data_path = data_path
        self.df = pd.DataFrame(columns=['repo', 'user', 'pr_title', 'pr_body', 'commits'])
        self.size = size
        self.pull_request_getter = PullRequestGetter('../token.txt')

    def generate_dataset(self):
        repo_names = self.get_repo_names()
        for repo in repo_names:
            print("Getting pulls for repo: " + repo)

            pulls = self.get_pulls_by_repo(repo)
            #Only taking top 100 pulls per repo
            pr_count = 0
            if pulls is None:
                continue
            for pr in pulls:
                if self.is_pr_valid(pr):
                    self.df = self.df.append(self.get_pr_data(pr, repo), ignore_index=True)
                    pr_count += 1
                    print("Pull request count for {}: {}".format(repo, pr_count))
                if pr_count >= 100:
                    break

            # if len(self.df) > self.size:
            #     break
            print("Finished getting pulls for repo: " + repo)
            self.df.to_csv(self.data_path + 'dataset7.csv', index=True)
        self.df.to_csv(self.data_path + 'fulldataset.csv', index=True)

    def get_repo_names(self, data_file='../data/repos.csv'):
        repos = pd.read_csv(data_file)
        #Header is rank,item,repo_name,stars,forks,language,repo_url,username,issues,last_commit,description
        #We want to return username + '/' + repo_name
        usernames = repos['username'].to_list()
        repo_names = repos['repo_name'].to_list()
        return [username + '/' + repo_name for username, repo_name in zip(usernames, repo_names)]

    def get_pulls_by_repo(self, repo):
        pulls = self.pull_request_getter.get_repo_pull_requests(repo)
        return pulls

    def parse_pull_body(self, body):
        #Replace urls in body with word LINK
        body = re.sub(r'https?://[^\s]+', 'LINK', body)
        #Replace images in body with word IMAGE
        body = re.sub(r'!\[.*\]\(.*\)', 'IMAGE', body)
        #Replace code blocks in body with word CODE
        body = re.sub(r'```.*```', 'CODE', body)
        return body

    def is_pr_valid(self, pr):
        #This could certainly be improved, to make for better filtering
        title = pr.title
        body = pr.body
        if title is None or body is None:
            return False
        if len(title) == 0 or len(body) == 0:
            return False
        #Update ReadMe.md is a common pr, but not a good one
        if 'update' in title.lower() and 'readme' in title.lower() and len(body) < 20:
            return False
        return True

    def get_pr_data(self, pr, repo_name):
        return {
            'repo': repo_name,
            'user': pr.user.login,
            'pr_title': pr.title,
            'pr_body': self.parse_pull_body(pr.body),
            'commits': pr.commits
        }


class ModelDatasetGenerator:
    def __init__(self, data_path='../data/', dataset_name='final_dataset_2.csv', size=1000000, split=0.5):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.df = pd.DataFrame(columns=['pr_title_1', 'pr_body_1', 'commits_1', 'pr_title_2', 'pr_body_2', 'commits_2',
                                        'same_repo', 'repo_1_name', 'repo_2_name', 'repo_1_user', 'repo_2_user'])
        self.size = size
        self.split = split

    def create_dataset(self, save_datapath='../data/'):
        df = pd.read_csv(self.data_path + self.dataset_name)
        df.drop(columns=['Unnamed: 0'], inplace=True)
        #We want to create a dataset comparing two samples from the original dataset, and whether they are from the same repo, with split being how often they are the same
        #Get random sample from original dataset, and with probability of split , get another sample from the same repo, else get another sample from a different repo
        #We want to do this for size number of times
        for i in range(self.size):
            sample_1 = df.sample(1)
            if random.random() < self.split:
                sample_2 = df[df['repo'] == sample_1['repo'].values[0]]
                sample_2 = sample_2.sample(1)
                same_repo = 1
            else:
                sample_2 = df[df['repo'] != sample_1['repo'].values[0]]
                sample_2 = sample_2.sample(1)
                same_repo = 0
            appending_row = {
                'pr_title_1': sample_1['pr_title'].values[0],
                'pr_body_1': sample_1['pr_body'].values[0],
                'commits_1': sample_1['commits'].values[0],
                'pr_title_2': sample_2['pr_title'].values[0],
                'pr_body_2': sample_2['pr_body'].values[0],
                'commits_2': sample_2['commits'].values[0],
                'same_repo': same_repo,
                'repo_1_name': sample_1['repo'].values[0],
                'repo_2_name': sample_2['repo'].values[0],
                'repo_1_user': sample_1['user'].values[0],
                'repo_2_user': sample_2['user'].values[0]
            }
            self.df = self.df.append(appending_row, ignore_index=True)
            if i % 1000 == 0:
                print('percent complete: ' + str((i / self.size)*100))
            if i % 10000 == 0:
                self.df.to_csv(save_datapath + 'combined_dataset_' + str(i) + '.csv')
                self.df = pd.DataFrame(
                    columns=['pr_title_1', 'pr_body_1', 'commits_1', 'pr_title_2', 'pr_body_2', 'commits_2',
                             'same_repo', 'repo_1_name', 'repo_2_name', 'repo_1_user', 'repo_2_user'])
        self.df.to_csv(self.data_path + 'combined_dataset.csv', index=True)



if __name__ == '__main__':
    model_dataset_generator = ModelDatasetGenerator()
    model_dataset_generator.create_dataset(save_datapath='../final_data/')