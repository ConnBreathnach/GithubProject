Do contributors to the same repo write commit messages in stylistically similar ways? Do programmers change their commit message style based on the repo that they are committing in?

Train a language model to on two commit messages, both by the same author in the same repo. Perhaps use some feature embedding and a dissimilarity measure. Simply training a binary classifier will not work in my opinion, since we also want to see if two different people have similar writing styles in their messages, and later we can see if one of these committers "adopted" the other user's style. Do repositories have a standard when writing commit messages?

Similarity analysis is generally used to detect if two sentences have a similar or same semantic meaning. This is no use to us since the same person could be working on two completely different sections of a codebase over time and therefore the commits may not have any semantic similarity.

Data gathering:
Use Github API to get all the commits for a repo. Save repo, commit message, and author. Only do so for repos with at least x committers (originally four), and at least 10 valid commits. Am only training on the top x repositories in python, so hopefully task will be less complex for model to analyse and train (fewer unique keywords, etc). Remove commits with redundant information (added README, merge branch into main, etc)

Potential issues: 
Machine learning is hard. Problem is not well defined (how do we measure dissimilarity. No baseline to go off of really). Commits will contain words relating to code written, so even very similar styles could be classified as very different based on keywords associated with the code implemented.

Training: Will train data on pairs of commits, correct ones will be by same author in same repo, incorrect will be two different authors in two different repos. The data will be gathered randomly (I do not have time to analyse thousands of commit messages), so this will certainly make the model less accurate.