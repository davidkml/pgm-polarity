import numpy as np
import torch
from torch.autograd import Variable
import pickle
from tqdm import tqdm

with open("feature_ori.p", 'rb') as f:
    data = pickle.load(f)

with open("label.p", 'rb') as f:
    labels = pickle.load(f)

data = np.array(data)
labels = np.array(labels)
ratio = len(data)*0.8
ratio = int(ratio)
indx = np.random.randint(0, len(data), ratio)
flags = np.zeros(len(data))
flags[indx] = 1
train_x, train_y = data[flags == 1], labels[flags == 1]
test_x, test_y = data[flags == 0], labels[flags == 0]



def test(net, data, label, batch_size):
    correct = 0
    total = 0
    net.eval()
    pbar = tqdm(range(0, data.shape[0], batch_size))
    for batch_num in pbar:
        total += 1
        if batch_num + batch_size > data.shape[0]:
            end = data.shape[0]
        else:
            end = batch_num + batch_size

        inputs_, actual_val = data[batch_num:end, :], label[batch_num:end]
        # perform classification
        inputs = torch.from_numpy(inputs_).float().cuda()

        predicted_val, embedding = net(torch.autograd.Variable(inputs))
        # convert 'predicted_val' GPU tensor to CPU tensor and extract the column with max_score
        predicted_val = predicted_val.data
        max_score, idx = torch.max(predicted_val, 1)
        assert idx.shape == actual_val.shape
        # compare it with actual value and estimate accuracy
        for i in range(idx.shape[0]):
            if idx[i] == actual_val[i]:
                correct += 1
        pbar.set_description("processing batch %s" % str(batch_num))
    print("Classifier Accuracy: ", correct/ data.shape[0])

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(200, 50)
        self.fc2 = torch.nn.Linear(50, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        embedding = self.fc1(x)
        out = self.relu(embedding)
        out = self.fc2(out)
        return out, embedding

learningRate = 0.01
epochs = 21
batch_size = 16

def train():
    model = MLP()
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    for epoch in range(epochs):
        # training set -- perform model training
        epoch_training_loss = 0.0
        num_batches = 0
        pbar = tqdm(range(0, train_x.shape[0], batch_size))
        for batch_num in pbar:  # 'enumerate' is a super helpful function
            # split training data into inputs and labels
            if batch_num+ batch_size> train_x.shape[0]:
                end = train_x.shape[0]
            else:
                end = batch_num+batch_size
            inputs_, labels_ =train_x[batch_num:end, :], train_y[batch_num:end]
            inputs = torch.from_numpy(inputs_).float().cuda()
            labels = torch.from_numpy(labels_).cuda()
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
            # Make gradients zero for parameters 'W', 'b'
            optimizer.zero_grad()
            forward_output, embedding = model(inputs)
            loss = criterion(forward_output, labels)
            loss.backward()
            optimizer.step()
            # calculating loss
            epoch_training_loss += loss.data.item()
            num_batches += 1
            # print(loss.data.item())
            pbar.set_description("processing batch %s" % str(batch_num))

        print("epoch: ", epoch, ", loss: ", epoch_training_loss / num_batches)
        test(model, test_x, test_y, batch_size=2000)
        if epoch%10 == 0:
            save_path = './cptk/model_' +str(epoch)+'.pth'
            torch.save(model.state_dict(), save_path)
