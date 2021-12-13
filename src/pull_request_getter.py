from github import Github

class PullRequestGetter:
    def __init__(self, token_file):
        try:
            with open(token_file, 'r') as f:
                self.token = f.read()
            self.github_api = Github(self.token)
        except FileNotFoundError:
            return None


    def get_repo_pull_requests(self, repo_name):
        try:
            repo = self.github_api.get_repo(repo_name)
        except Exception as e:
            return "Repo not found"
        repo_contributors = repo.get_contributors()
        if repo_contributors.totalCount < 4:
            return None
        pulls = repo.get_pulls(state='all')  # Interested in all PRs, closed are in fact better as they are often the ones that have been merged
        if pulls.totalCount < 10:
            return None
        return pulls
