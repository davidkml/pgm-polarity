import csv
import heapq
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

if __name__ == '__main__':
    
    node_label = {}
    word_freq = {}
    #read node labels
    with open('data/icwsm_polarization/all.nodes') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            # 1 for left, 0 for right
            if row[1] == 'left':
                node_label[row[0]] = 1
            elif row[1] == 'right':
                node_label[row[0]] = 0

    #read hashtags
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
        
        most_freq = heapq.nlargest(200, word_freq, key=word_freq.get)

        #build X and y dataset
        X_list = []
        y_list = []
        #iterate over every row in all.edgelist
        csv_file.seek(0)
        for row in csv_reader:
            datapoint = []
            #if retweet, take label from B node, else take label from A node
            if row[2] == 'retweet':
                #check if B node in edge is in labels
                if row[1] not in node_label.keys():
                    continue
                y_list.append(node_label[row[1]])
                for token in most_freq:
                    if token in row[5:]:
                        datapoint.append(1)
                    else:
                        datapoint.append(0)
                X_list.append(datapoint)
            else:
                if row[0] not in node_label.keys():
                    continue
                y_list.append(node_label[row[0]])
                for token in most_freq:
                    if token in row[5:]:
                        datapoint.append(1)
                    else:
                        datapoint.append(0)
                X_list.append(datapoint)
        
        X = np.asarray(X_list)
        y = np.asarray(y_list)


    print("Unique hashtags: {}".format(len(word_freq)))
    print("Total number of edges: {}".format(line_count))
    print(f'Total number of labels: {len(node_label)}')
    print(f'Shape of X: {X.shape}')
    print(f'Shape of y: {y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)

    #Train the model using the training sets
    knn.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = knn.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))