from pathlib import Path
import pandas as pd
from collections import defaultdict
import csv
import heapq
import numpy as np
from pystruct.models import GraphCRF
from pystruct.learners import FrankWolfeSSVM
import pickle
from combine_utils import *
import torch
import igraph


class CrfClassifier:
    def __init__(self):
        self.top_seq = 200
        self.embedding_dim = 50
<<<<<<< HEAD
=======
        self.pert_dict = {}
>>>>>>> 93309e3207d37152eefafa6b563c72777a863935

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
                cnt += 1

            total_datas.append((features, edge_nodes.astype(int)))
            total_labels.append(clique_labels)

        ratio = len(total_datas)*0.7
        ratio = int(ratio)
        print(ratio)
        X_train, y_train = total_datas[:ratio], total_labels[:ratio]
        X_test, y_test = total_datas[ratio:], total_labels[ratio:]

        model = GraphCRF(inference_method="max-product")
        ssvm = FrankWolfeSSVM(model=model, C=0.1, max_iter=10)
        ssvm.fit(X_train, y_train)
        result = ssvm.score(X_test, y_test)
        print(result)

    def get_datas(self, c_idxs, labels, mentions, retweets, bags):
        train_datas = []
        train_labels = []
        node_ids = []
        for user_id in c_idxs:
            user_id = str(user_id)
            if not user_id in labels:
                continue
            if not user_id in bags:
                continue
            bag = bags[user_id]
            features = np.empty((0, self.top_seq))
            edge_nodes = np.empty((0, 2))
            node_id = np.empty((0, 1))
            clique_labels = np.array([labels[user_id]])
            features = np.vstack([features, bag])
            node_id = np.vstack([node_id, np.array([[int(user_id)]])])
            mentioned_ids = mentions[user_id]
            cnt = 0
            for mentioned_id in mentioned_ids:
                if not int(mentioned_id) in c_idxs:
                    continue
                if not mentioned_id in labels:
                    continue
                if not mentioned_id in bags:
                    continue
                clique_labels = np.append(clique_labels, np.array([labels[mentioned_id]]))
                if mentioned_id in bags:
                    features = np.vstack([features, bags[mentioned_id]])
                else:
                    features = np.vstack([features, np.zeros(self.top_seq)])
                node_id = np.vstack([node_id, np.array([[int(mentioned_id)]])])
                edge_nodes = np.vstack([edge_nodes, np.array([0, cnt + 1])])
                cnt += 1

            num_mentioned = edge_nodes.shape[0]
            retweet_ids = retweets[user_id]
            cnt = 0
            for retweet_id in retweet_ids:
                if not int(retweet_id) in c_idxs:
                    continue
                if not retweet_id in labels:
                    continue
                if not retweet_id in bags:
                    continue
                clique_labels = np.append(clique_labels, np.array([labels[retweet_id]]))
                if retweet_id in bags:
                    features = np.vstack([features, bags[retweet_id]])
                else:
                    features = np.vstack([features, np.zeros(self.top_seq)])
                node_id = np.vstack([node_id, np.array([[int(retweet_id)]])])
                edge_nodes = np.vstack([edge_nodes, np.array([0, cnt + 1 + num_mentioned])])
                cnt += 1
<<<<<<< HEAD

=======
>>>>>>> 93309e3207d37152eefafa6b563c72777a863935
            train_datas.append((features, edge_nodes.astype(int)))
            train_labels.append(clique_labels)
            node_ids.append(node_id)
        return train_datas, train_labels, node_ids


    def combined_trainng(self, pert, bags, mentions, retweets, labels):
        print('load graph')
        all_graph = pickle.load(open('./data/icwsm_polarization/all_igraph.pickle', "rb"))
        adj_matrix = np.load("objects/adj_matrix.npz")['arr_0']
        print('sampling labels')
        clabels, c_idxs = get_labels_subsample(all_graph, 'centrality', pert)
<<<<<<< HEAD
        adj_matrix_t = torch.FloatTensor(adj_matrix)
        test_ids_ori = np.where(clabels == -1)[0]

        # training with label propagation
=======
        test_ids_ori = np.where(clabels == -1)[0]
        adj_matrix_t = torch.FloatTensor(adj_matrix)

        # training with label propagation
        test_datas, test_labels, node_ids = self.get_datas(range(0, clabels.shape[0]), labels, mentions, retweets, bags)
        self.pert_dict[pert] = []
>>>>>>> 93309e3207d37152eefafa6b563c72777a863935
        for i in range(10):
            clabels_t = torch.LongTensor(clabels)
            label_propagation_central = LabelPropagation(adj_matrix_t)
            label_propagation_central.fit(clabels_t)
            label_propagation_output_labels_central = label_propagation_central.predict_classes()
            central_propagation_df = pd.DataFrame(data=label_propagation_output_labels_central.numpy(), columns=["label"])
            test_ids = np.where(clabels == -1)[0]
            # training with CRF
            print('find training data')
            train_datas, train_labels, _ = self.get_datas(c_idxs, labels, mentions, retweets, bags)
<<<<<<< HEAD
            test_datas, test_labels, node_ids = self.get_datas(test_ids, labels, mentions, retweets, bags)
            if i == 0:
                x_test_ori, y_test_ori = test_datas, test_labels
=======
>>>>>>> 93309e3207d37152eefafa6b563c72777a863935
            print(len(train_datas))
            print(len(test_datas))
            X_train, y_train = train_datas, train_labels

            model = GraphCRF(inference_method="max-product")
            ssvm = FrankWolfeSSVM(model=model, C=0.1, max_iter=10)
            ssvm.fit(X_train, y_train)
            y_preds = ssvm.predict(test_datas)
<<<<<<< HEAD
            result = ssvm.score(x_test_ori, y_test_ori)
            print('iter {} result = {}'.format(i, result))
            count = 0
            for clique_idx, clique in enumerate(y_preds):
                for node_idx, node in enumerate(clique):
                    node_id = node_ids[clique_idx][node_idx]
                    if node == central_propagation_df.iloc[node_id].values:
                        clabels[int(node_id)] = node
                        if not int(node_id) in c_idxs:
                            c_idxs = np.append(c_idxs, int(node_id))
                            count += 1
            print('iter {} update {} new labels'.format(i, count))
=======
            # result = ssvm.score(test_datas, test_labels)
            # print('iter {} result = {}'.format(i, result))
            count = 0
            accuracy = []
            total = []
            for clique_idx, clique in enumerate(y_preds):
                for node_idx, node in enumerate(clique):
                    node_id = node_ids[clique_idx][node_idx][0]
                    node_id = int(node_id)
                    if node_id in test_ids_ori and str(node_id) in labels:
                        if node_id not in total:
                            total.append(node_id)
                        if node == labels[str(node_id)]:
                            if not node_id in accuracy:
                                accuracy.append(node_id)

                    if node_id in test_ids:
                        if node == central_propagation_df.iloc[node_id].values:
                            clabels[int(node_id)] = node
                            if not int(node_id) in c_idxs:
                                c_idxs = np.append(c_idxs, int(node_id))
                                count += 1
            print('iter {} update {} new labels'.format(i, count))
            result = len(accuracy)/float(len(total))
            print('iter {} result = {}/{} = {}'.format(i, len(accuracy), len(total), result))
            self.pert_dict[pert].append([result, count])
            if count == 0:
               break
>>>>>>> 93309e3207d37152eefafa6b563c72777a863935




if __name__ == '__main__':
    crf = CrfClassifier()
    labels = crf.get_labels()
    mentions, retweets, user_tags = crf.establish_dict_using_all()

    with open("./objects/pca200_df.pickle", 'rb') as f:
        bags = pickle.load(f)
    pca_dict= {}
    for i in range(bags.shape[0]):
        pca_dict[str(i)] = bags.iloc[i].values
    # crf.structraining(pca_dict, mentions, retweets, labels)

<<<<<<< HEAD
    crf.combined_trainng(0.5, pca_dict, mentions, retweets, labels)

=======
    for pert in np.arange(0.1, 1, 0.1):
        print('training with {} seen........................................'.format(pert))
        crf.combined_trainng(pert, pca_dict, mentions, retweets, labels)

    with open('pert_dict.p','wb') as f:
        pickle.dump(crf.pert_dict, f)
>>>>>>> 93309e3207d37152eefafa6b563c72777a863935

