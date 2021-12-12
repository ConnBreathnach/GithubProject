import unittest
from src.pull_request_getter import PullRequestGetter

class TestPRGetter(unittest.TestCase):

    #Need to fix this test, but running into issues when raising exceptions from constructor
    # def test_constructor(self):
    #     self.assertRaises(FileNotFoundError, PullRequestGetter('this_is_an_invalid_token'))

    def test_pr_getter(self):
        pr_getter = PullRequestGetter('../token.txt')
        self.assertEquals(pr_getter.get_repo_pull_requests("NIAOEGNIGAEONIOGERNIOAFHBUBUIOAEBGUISBWIOAHIOPGEIP"), "Repo not found", "Testing on invalid repo name")
        self.assertIsNone(pr_getter.get_repo_pull_requests("ConnBreathnach/MakingUsers"), "Testing on repository with less than four contributors")
        self.assertIsNotNone(pr_getter.get_repo_pull_requests("PyGithub/PyGithub"), "Testing on working repository, PyGithub in this case")



if __name__ == '__main__':
    unittest.main()
