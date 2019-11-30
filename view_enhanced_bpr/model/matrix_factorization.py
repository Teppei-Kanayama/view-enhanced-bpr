import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd


class MatrixFactorization(nn.Module):
    def __init__(self, n_items, n_users, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding_layer = nn.Embedding(num_embeddings=n_users, embedding_dim=embedding_dim)
        self.item_embedding_layer = nn.Embedding(num_embeddings=n_items, embedding_dim=embedding_dim)

    def forward(self, user, item):
        user = self.user_embedding_layer(user)
        item = self.item_embedding_layer(item)
        return (user * item).sum(axis=1)

    @staticmethod
    def data_sampler(data, batch_size=2**11, iterations=1000000):
        for i in range(0, iterations):
            batch = data.sample(batch_size)
            yield dict(
                user_indices=Variable(torch.FloatTensor(batch['user_index'].values)).long(),
                clicked_item_indices=Variable(torch.FloatTensor(batch['clicked_item_index'].values)).long(),
                not_clicked_item_indices=Variable(torch.FloatTensor(batch['not_clicked_item_index'].values)).long()
                )
