import statistics
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime

import pandas as pd
import numpy as np

from logging import getLogger

import gokart
import luigi

from view_enhanced_bpr.data.preprocess_data import PreprocessData

logger = getLogger(__name__)


class TrainModel(gokart.TaskOnKart):
    task_namespace = 'view_enhanced_bpr'

    def requires(self):
        return PreprocessData()

    def run(self):
        data = self.load()['train']



        import pdb; pdb.set_trace()

        movie_index = json.load(open('./works/defs/smovie_index.json'))
        user_index = json.load(open('./works/defs/user_index.json'))
        print('user size', len(user_index), 'item size', len(movie_index))
        model = MF(len(movie_index), len(user_index), embedding_dim=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        for index, (uindex, mindex, scores) in enumerate(generate()):
            uindex_t = Variable(torch.FloatTensor(uindex)).long()
            mindex_t = Variable(torch.FloatTensor(mindex)).long()
            predict = model([mindex_t, uindex_t])

            scores = Variable(torch.FloatTensor(scores)).float()

            loss = nn.MSELoss()(predict, scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if index % 500 == 0:
                print(index, datetime.now())
                validate(model)


class MF(nn.Module):
    def __init__(self, input_items, input_users, embedding_dim):
        super(MF, self).__init__()
        self.l_b1 = nn.Embedding(num_embeddings=input_items, embedding_dim=embedding_dim)
        self.l_a1 = nn.Embedding(num_embeddings=input_users, embedding_dim=embedding_dim)
        self.l_l1 = nn.Linear(in_features=embedding_dim, out_features=1, bias=True)

    def forward(self, inputs):
        item_vec, user_vec = inputs
        item_vec = self.l_b1(item_vec)
        user_vec = self.l_a1(user_vec)
        return F.relu(self.l_l1(user_vec * item_vec))


def myLoss(output, target):
    loss = torch.sqrt(torch.mean((output-target)**2))
    return loss


def generate():
    train_triples = pickle.load(open('works/dataset/train_triples.pkl', 'rb'))
    BATCH = 32
    for i in range(0, len(train_triples), BATCH):
        array = np.array(train_triples[i:i+BATCH])
        uindex = array[:, 0]
        mindex = array[:, 1]
        scores = array[:, 2]
        yield uindex, mindex, scores


# test_triples = pickle.load(open('works/dataset/test_triples.pkl', 'rb'))


def validate(model):
    array = np.array(test_triples)
    losses = []
    for i in range(0, array.shape[0], 100):
        uindex = array[i:i+100, 0]
        mindex = array[i:i+100, 1]
        scores = array[i:i+100, 2]
        inputs = [
            Variable(torch.FloatTensor(mindex)).long(),
            Variable(torch.FloatTensor(uindex)).long(),
        ]
        scores = Variable(torch.FloatTensor(scores)).float()
        loss = myLoss(scores, model(inputs))
        losses.append(float(loss.data.numpy()))
        del inputs
    print('rmse', statistics.mean(losses))
