import statistics
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime

from logging import getLogger

import gokart
import luigi

from view_enhanced_bpr.data.preprocess_data import PreprocessData

logger = getLogger(__name__)


class TrainModel(gokart.TaskOnKart):
    task_namespace = 'view_enhanced_bpr'

    # random_state = luigi.IntParameter(default=615)  # type: # int
    validation_ratio = luigi.FloatParameter(default=0.1)  # type: float

    def requires(self):
        return PreprocessData()

    def run(self):
        data = self.load()['train']
        n_users = data['user_index'].max() + 1
        n_items = data['item_index'].max() + 1

        validation_data = data[
            (data['user_index'] > n_users * (1 - self.validation_ratio)) & (data['item_index'] > n_items * (1 - self.validation_ratio))]

        train_data = data.drop(validation_data.index)

        model = MF(n_items, n_users, embedding_dim=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)

        for iterations, (user_index, item_index, scores) in enumerate(model.data_sampler(train_data)):
            user_tensor = Variable(torch.FloatTensor(user_index)).long()
            item_tensor = Variable(torch.FloatTensor(item_index)).long()
            scores = Variable(torch.FloatTensor(scores)).float()

            predict = model([item_tensor, user_tensor])
            loss = nn.MSELoss()(predict, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iterations % 1000 == 0:
                # print(iterations, datetime.now())
                print(validate(model, validation_data))


class MF(nn.Module):
    def __init__(self, input_items, input_users, embedding_dim):
        super(MF, self).__init__()
        self.l_b1 = nn.Embedding(num_embeddings=input_items, embedding_dim=embedding_dim)
        self.l_a1 = nn.Embedding(num_embeddings=input_users, embedding_dim=embedding_dim)
        # self.l_l1 = nn.Linear(in_features=embedding_dim, out_features=1, bias=True)

    def forward(self, inputs):
        item_vec, user_vec = inputs
        item_vec = self.l_b1(item_vec)
        user_vec = self.l_a1(user_vec)
        return (user_vec * item_vec).sum(axis=1)
        # return F.relu(self.l_l1(user_vec * item_vec))

    @staticmethod
    def data_sampler(data, batch_size=2**11, iterations=1000000):
        data = data[data['rating'] > 0]
        for i in range(0, iterations):
            batch = data.sample(batch_size)
            user_indices = batch['user_index'].values
            item_indices = batch['item_index'].values
            scores = batch['rating'].values
            yield user_indices, item_indices, scores


def validate(model, data):
    user_tensor = Variable(torch.FloatTensor(data['user_index'].values)).long()
    item_tensor = Variable(torch.FloatTensor(data['item_index'].values)).long()
    scores = model([item_tensor, user_tensor])
    data['model_score'] = scores.data.numpy()
    data['rank'] = data.groupby('user')['model_score'].rank(ascending=False)

    gt_clicks = data.groupby('user', as_index=False).agg({'click': 'sum'}).rename(columns={'click': 'gt_clicks'})
    gt_clicks = gt_clicks[gt_clicks['gt_clicks'] > 0]

    model_clicks = data[data['rank'] <= 10].groupby('user', as_index=False).agg({'click': 'sum'}).rename(columns={'click': 'model_clicks'})
    clicks = pd.merge(gt_clicks, model_clicks, on='user', how='left')
    clicks['recall'] = clicks['model_clicks'] / clicks['gt_clicks']

    return clicks['recall'].mean()


# def validate(model, data):
#     array = data[['user_index', 'item_index', 'rating']].values
#     losses = []
#     for i in range(0, array.shape[0], 100):
#         uindex = array[i:i+100, 0]
#         mindex = array[i:i+100, 1]
#         scores = array[i:i+100, 2]
#         inputs = [
#             Variable(torch.FloatTensor(mindex)).long(),
#             Variable(torch.FloatTensor(uindex)).long(),
#         ]
#         scores = Variable(torch.FloatTensor(scores)).float()
#         loss = validation_loss(scores, model(inputs))
#         losses.append(float(loss.data.numpy()))
#         del inputs
#     print('rmse', statistics.mean(losses))


# def validation_loss(output, target):
#     loss = torch.sqrt(torch.mean((output-target)**2))
#     return loss
