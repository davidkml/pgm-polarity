from pathlib import Path
import pandas as pd
from collections import defaultdict
import csv
import heapq
import numpy as np
from sklearn_crfsuite import metrics
import sklearn_crfsuite
from pystruct.models import GraphCRF, EdgeFeatureGraphCRF
from pystruct.learners import FrankWolfeSSVM
import pickle
from feature_embedding import MLP
import torch

class CrfClassifier:
    def __init__(self):
        self.top_seq = 200
        self.embedding_dim = 50

    def establish_vocabulary(self):
        word_freq = {}
        with open('data/icwsm_polarization/all.edgelist') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            line_count = 0
            for row in csv_reader:
                for token in row[5:]:
                    if token not in word_freq.keys():
                        word_freq[token] = 1
                    else:
                        word_freq[token] += 1
                line_count += 1

            most_freq = heapq.nlargest(self.top_seq, word_freq, key=word_freq.get)
        return most_freq

    def get_labels(self):
        node_label = {}
        count = 0
        with open('data/icwsm_polarization/all.nodes') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                # 1 for left, 0 for right
                if row[1] == 'left':
                    node_label[row[0]] = 1
                elif row[1] == 'right':
                    node_label[row[0]] = 0
                else:
                    count += 1
        print('{} people not have labels'.format(count))
        return node_label

    def establish_dict(self):
        mention_file_path = Path('data/icwsm_polarization/mention.edgelist')
        retweet_file_path = Path('data/icwsm_polarization/retweet.edgelist')

        mention_edgelist = pd.read_csv(mention_file_path).values.tolist()
        retweet_edgelist = pd.read_csv(retweet_file_path).values.tolist()

        mentions, retweets, user_tags = defaultdict(list), defaultdict(list), defaultdict(list)

        for m in mention_edgelist:
            curr = m[0].split()
            tags = curr[5:]
            mentions[curr[0]].append(curr[1])
            user_tags[curr[0]] = tags

        # print(mentions)
        # print(user_tags)  # tags can have duplicates for now since we just iterate through everything

        for r in retweet_edgelist:
            curr = r[0].split()
            tags = curr[5:]
            retweets[curr[1]].append(curr[0])
            user_tags[curr[1]].extend(tags)

        # print(retweets)
        # print(user_tags)  # each user has tags from mentions & retweets
        return mentions, retweets, user_tags

    def establish_dict_using_all(self):
        all_file_path = Path('data/icwsm_polarization/all.edgelist')

        all_edgelist = pd.read_csv(all_file_path).values.tolist()

        mentions, retweets, user_tags = defaultdict(list), defaultdict(list), defaultdict(list)

        for m in all_edgelist:
            curr = m[0].split()
            tags = curr[5:]
            if curr[2] == 'retweet':
                retweets[curr[1]].append(curr[0])
                if not curr[1] in user_tags:
                    user_tags[curr[1]] = tags
                else:
                    user_tags[curr[1]].extend(tags)
            if curr[2] == 'reply':
                mentions[curr[0]].append(curr[1])
                if not curr[0] in user_tags:
                    user_tags[curr[0]] = tags
                else:
                    user_tags[curr[0]].extend(tags)
        return mentions, retweets, user_tags

    def establish_bag(self, user_tags, vocabulary):
        bag = dict()
        for user_id, tags in user_tags.items():
            user_bag = np.zeros(len(vocabulary))
            for tag in tags:
                if tag in vocabulary:
                    word_id  = vocabulary.index(tag)
                    user_bag[word_id] += 1
            bag[user_id] = user_bag
        return bag

    def node2feature(self, bags, mentions, retweets, labels):
        users = bags.keys()
        total_labels = []
        total_features = []
        count = 0
        for user in users:
            if not user in labels:
                count += 1
                continue
            feature = {}
            bag = bags[user]
            for j in range(len(bag)):
                feature['bag{}'.format(j)] = bag[j]
            mention_users = mentions[user]

            for i in range(10):
                if i < len(mention_users):
                    if mention_users[i] in bags:
                        bag = bags[mention_users[i]]
                        for j in range(len(bag)):
                            feature['mention{}{}'.format(i, j)] = bag[j]

            retweet_users = retweets[user]
            for i in range(10):
                if i < len(retweet_users):
                    if retweet_users[i] in bags:
                        bag = bags[retweet_users[i]]
                        for j in range(len(bag)):
                            feature['retweet{}{}'.format(i, j)] = bag[j]

            total_features.append([feature])
            total_labels.append([labels[user]])
        print('{} people not include in features'.format(count))
        return total_features, total_labels

    def node2featurev2(self, bags, mentions, retweets, user_tags):
        users = bags.keys()
        total_labels = []
        total_features = []
        for user in users:
            if not user in labels:
                print('user doesnt have label')
                continue
            feature = {}
            tags = user_tags[user]
            for j in range(len(tags)):
                feature['tag{}'.format(j)] = tags[j]
            mention_users = mentions[user]

            for i in range(10):
                if i < len(mention_users):
                    if mention_users[i] in bags:
                        tags = user_tags[mention_users[i]]
                        for j in range(len(tags)):
                            feature['tag{}'.format(j)] = tags[j]

            retweet_users = retweets[user]
            for i in range(10):
                if i < len(retweet_users):
                    if retweet_users[i] in bags:
                        tags = user_tags[retweet_users[i]]
                        for j in range(len(tags)):
                            feature['tag{}'.format(j)] = tags[j]

            total_features.append([feature])
            total_labels.append([labels[user]])
        return total_features, total_labels

    def suitetraining(self, features, labels):
        ratio = len(features) / 2
        ratio = int(ratio)

        train_x, train_y = features[:ratio], labels[:ratio]
        test_x, test_y = features[ratio:], labels[ratio:]

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(train_x, train_y)
        labels = list(crf.classes_)

        y_pred = crf.predict(test_x)

        result = metrics.flat_f1_score(test_y, y_pred,
                                       average='weighted', labels=labels)
        print(result)

    def structraining(self, bags, mentions, retweets, labels):
        total_datas = []
        total_labels = []
        print('num_user', len(bags.keys()))
        for user_id, bag in bags.items():
            if not user_id in labels:
                continue
            features = np.empty((0, self.top_seq))
            edge_nodes = np.empty((0, 2))
            edge_features = np.empty((0, 1))
            clique_labels = np.array([labels[user_id]])
            features = np.vstack([features, bag])
            mentioned_ids = mentions[user_id]
            cnt = 0
            for mentioned_id in enumerate(mentioned_ids):
                if not mentioned_id in labels:
                    continue
                clique_labels = np.append(clique_labels, np.array([labels[mentioned_id]]))
                if mentioned_id in bags:
                    features = np.vstack([features, bags[mentioned_id]])
                else:
                    features = np.vstack([features, np.zeros(self.top_seq)])
                edge_nodes = np.vstack([edge_nodes, np.array([0, cnt + 1])])
                edge_features = np.vstack([edge_features, np.array([[0]])])
                cnt += 1

            num_mentioned = edge_nodes.shape[0]
            retweet_ids = retweets[user_id]
            cnt = 0
            for retweet_id in retweet_ids:
                if not retweet_id in labels:
                    continue
                clique_labels = np.append(clique_labels, np.array([labels[retweet_id]]))
                if retweet_id in bags:
                    features = np.vstack([features, bags[retweet_id]])
                else:
                    features = np.vstack([features, np.zeros(self.top_seq)])
                edge_nodes = np.vstack([edge_nodes, np.array([0, cnt + 1 + num_mentioned])])
                edge_features = np.vstack([edge_features, np.array([[1]])])
                cnt += 1

            total_datas.append((features, edge_nodes.astype(int), edge_features))
            total_labels.append(clique_labels)

        ratio = len(total_datas)*0.7
        ratio = int(ratio)
        print(ratio)
        X_train, y_train = total_datas[:ratio], total_labels[:ratio]
        X_test, y_test = total_datas[ratio:], total_labels[ratio:]

        model = EdgeFeatureGraphCRF(inference_method="max-product")
        ssvm = FrankWolfeSSVM(model=model, C=0.1, max_iter=10)
        ssvm.fit(X_train, y_train)
        result = ssvm.score(X_test, y_test)
        print(result)

    def to_train_embedding(self, bags, labels):
        total_features = []
        total_labels = []
        for user_id, bag in bags.items():
            if user_id in labels:
                total_features.append(bag)
                total_labels.append(labels[user_id])

        with open('feature_ori.p', 'wb') as f:
            pickle.dump(total_features, f)

        with open('label.p', 'wb') as f:
            pickle.dump(total_labels, f)

    def extract_feature(self, data):
        net = MLP()
        net.load_state_dict(torch.load('./cptk/model_20.pth'))
        ##### For GPU #######
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
        feature_dict = {}
        for user_id, bag in data.items():
            input = torch.from_numpy(bag).float().cuda()
            _, embedding = net(torch.autograd.Variable(input))
            feature_dict[user_id] = embedding.data.cpu().numpy()

        with open('embedding_feature.p', 'wb') as f:
            pickle.dump(feature_dict, f)

    def embedding_training(self, mentions, retweets, labels):
        with open('embedding_feature.p', 'rb') as f:
            bags = pickle.load(f)

        print('num_user', len(bags.keys()))
        total_datas = []
        total_labels = []
        for user_id, bag in bags.items():
            if not user_id in labels:
                continue
            features = np.empty((0, self.embedding_dim))
            edge_nodes = np.empty((0, 2))
            edge_features = np.empty((0, 1))
            clique_labels = np.array([labels[user_id]])
            features = np.vstack([features, bag])
            mentioned_ids = mentions[user_id]
            cnt = 0
            for mentioned_id in enumerate(mentioned_ids):
                if not mentioned_id in labels:
                    continue
                clique_labels = np.append(clique_labels, np.array([labels[mentioned_id]]))
                if mentioned_id in bags:
                    features = np.vstack([features, bags[mentioned_id]])
                else:
                    features = np.vstack([features, np.zeros(self.embedding_dim)])
                edge_nodes = np.vstack([edge_nodes, np.array([0, cnt + 1])])
                edge_features = np.vstack([edge_features, np.array([[0]])])
                cnt += 1

            num_mentioned = edge_nodes.shape[0]
            retweet_ids = retweets[user_id]
            cnt = 0
            for retweet_id in retweet_ids:
                if not retweet_id in labels:
                    continue
                clique_labels = np.append(clique_labels, np.array([labels[retweet_id]]))
                if retweet_id in bags:
                    features = np.vstack([features, bags[retweet_id]])
                else:
                    features = np.vstack([features, np.zeros(self.embedding_dim)])
                edge_nodes = np.vstack([edge_nodes, np.array([0, cnt + 1 + num_mentioned])])
                edge_features = np.vstack([edge_features, np.array([[1]])])
                cnt += 1

            total_datas.append((features, edge_nodes.astype(int), edge_features))
            total_labels.append(clique_labels)

        ratio = len(total_datas) * 0.7
        ratio = int(ratio)
        print(ratio)
        X_train, y_train = total_datas[:ratio], total_labels[:ratio]
        X_test, y_test = total_datas[ratio:], total_labels[ratio:]

        model = EdgeFeatureGraphCRF(inference_method="max-product")
        ssvm = FrankWolfeSSVM(model=model, C=0.1, max_iter=10)
        ssvm.fit(X_train, y_train)
        result = ssvm.score(X_test, y_test)
        print(result)


if __name__ == '__main__':
    crf = CrfClassifier()
    labels = crf.get_labels()
    mentions, retweets, user_tags = crf.establish_dict_using_all()
    vocabulary = crf.establish_vocabulary()
    assert  len(vocabulary) == crf.top_seq
    bags = crf.establish_bag(user_tags, vocabulary)
    # crf.to_train_embedding(bags, labels)
    # crf.extract_feature(bags)
    # crf.embedding_training(mentions, retweets, labels)
    # for key in bags.keys():
    #     print(bags[key])
    # features, labels = crf.node2feature(bags, mentions, retweets, labels)
    # crf.suitetraining(features, labels)
    # with open("./objects/pca200_df.pickle", 'rb') as f:
    #     bags = pickle.load(f)
    pca_dict= {}
    for i in range(bags.shape[0]):
        pca_dict[str(i)] = bags.iloc[i].values
    crf.structraining(pca_dict, mentions, retweets, labels)



