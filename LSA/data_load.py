from sklearn.datasets import fetch_20newsgroups


newsgroups_train = fetch_20newsgroups()
data = newsgroups_train.data
target = newsgroups_train.target
