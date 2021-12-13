import pandas as pd
from pull_request_getter import PullRequestGetter
from github import Github
import re

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
            #Only taking top 1000 pulls per repo
            pr_count = 0
            for pr in pulls:
                if self.is_pr_valid(pr):
                    self.df = self.df.append(self.get_pr_data(pr, repo), ignore_index=True)
                    pr_count += 1
                    print("Pull request count for {}: {}".format(repo, pr_count))
                if pr_count >= 1000:
                    break
            # if len(self.df) > self.size:
            #     break
            print("Finished getting pulls for repo: " + repo)
        self.df.to_csv(self.data_path + 'dataset.csv', index=False)

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
        return [repo_name, pr.user.login, pr.title, self.parse_pull_body(pr.body), pr.commits]



if __name__ == '__main__':
    dataset_generator = DatasetGenerator()
    dataset_generator.generate_dataset()