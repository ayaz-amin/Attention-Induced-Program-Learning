import copy

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam

from forward_model.model import ConditionalAttentionInduction
from inference_model.learning import train_image
from inference_model.inference import test_image


def classification_run(folder, n_iters=1000):
    root = 'experiments/omniglot/all_runs'
    file_path = root+'/'+folder
    fname_label = 'class_labels.txt'

    # get file names
    with open(file_path+'/'+fname_label) as f:
        content = f.read().splitlines()
    pairs = [line.split() for line in content]
    test_files  = [pair[0] for pair in pairs]
    train_files = [pair[1] for pair in pairs]
    answers_files = copy.copy(train_files)

    test_files.sort()
    train_files.sort()	
    
    ntrain = len(train_files)
    ntest = len(test_files)

    # load the images (and, if needed, extract features)
    train_dataset, labels = load_dataset(root, train_files)
    test_items, _ = load_dataset(root, test_files)
    
    model_factors, labels_ = train_model(train_dataset, labels, n_iters)

    costM = []
    for i in range(ntest):
        img = test_items[i].numpy()
        class_idx, _ = test_image(img, model_factors)
        costM.append(labels_[class_idx])
	
    # compute the error rate
    correct = 0.0
    for i in range(ntest):
        if train_files[costM[i]] == answers_files[i]:
            correct += 1.0
    pcorrect = 100 * correct / ntest
    perror = 100 - pcorrect
    return perror


def load_dataset(file_path, dataset):
    data_array = []
    class_index = []

    for i, (file) in enumerate(dataset):
        full_path = file_path+'/'+file
        image = cv2.imread(full_path, 0)
        image = cv2.bitwise_not(image)
        image = torch.from_numpy(image)
        data_array.append(image)
        class_index.append(i)

    return data_array, class_index


def train_model(dataset, labels, n_iters):
    rcn_model_set = []
    rcn_label_set = []

    model = ConditionalAttentionInduction(num_classes=len(labels))
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    model.train()

    for class_index in labels:
        mean_loss = 0
        noise = torch.randn((105, 105))
        for _ in range(n_iters):
            optimizer.zero_grad()
            image = dataset[class_index].float()
            pimg = model(class_index, noise)
            loss = F.mse_loss(pimg, image)
            mean_loss += loss
            loss.backward()
            optimizer.step()

        print(mean_loss / n_iters)

    for class_index in labels:
        for _ in range(5):
            sample = model(class_index).detach().numpy()
            rcn_model_set.append(train_image(sample))
            rcn_label_set.append(class_index)
    
    return list(zip(*rcn_model_set)), rcn_label_set

    
if __name__ == "__main__":

    print('One Shot Classification')

    perror = np.zeros(20)
    for r in range(1, 21):
        rs = str(r)
        if len(rs)==1:
            rs = '0' + rs		
        perror[r-1] = classification_run('run'+rs)
        print(" run " + str(r) + " (error " + str(	perror[r-1] ) + "%)")		
    total = np.mean(perror)
    print(" average error " + str(total) + "%")