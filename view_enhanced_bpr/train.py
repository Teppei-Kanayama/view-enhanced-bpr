import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import gokart
import luigi

from view_enhanced_bpr.data.preprocess_data import PreprocessData, MakeTrainPair
from view_enhanced_bpr.model.matrix_factorization import MatrixFactorization
from view_enhanced_bpr.model.loss import bpr_loss, view_enhanced_bpr_loss


class TrainModel(gokart.TaskOnKart):
    task_namespace = 'view_enhanced_bpr'
    validation_ratio = luigi.FloatParameter(default=0.1)  # type: float
    embedding_dim = luigi.IntParameter(default=10)  # type: int
    lr = luigi.FloatParameter(default=0.005)  # type: float
    weight_decay = luigi.FloatParameter(default=0.0001)  # type: float
    alpha = luigi.FloatParameter(default=0.5)  # type: float
    loss_type = luigi.Parameter(default='bpr')  # type: str

    def requires(self):
        return dict(data=PreprocessData(), train_data=MakeTrainPair())

    def output(self):
        return self.make_model_target(relative_file_path='model/mf.zip',
                                      save_function=torch.save,
                                      load_function=torch.load)

    def run(self):
        data = self.load('data')['train']
        n_users = data['user_index'].max() + 1
        n_items = data['item_index'].max() + 1

        train_pair_data = self.load('train_data')
        validation_data = self.load('data')['validation']

        model = MatrixFactorization(n_items=n_items, n_users=n_users, embedding_dim=self.embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        loss_functions = dict(
            bpr=bpr_loss,
            view_enhanced_bpr=view_enhanced_bpr_loss
            )
        loss_function = loss_functions[self.loss_type]

        data = model.data_sampler(train_pair_data)

        training_losses = []
        for iterations, d in enumerate(data):
            if self.loss_type == 'bpr':
                predict = [model(item=d['clicked_item_indices'], user=d['user_indices']),
                           model(item=d['not_clicked_item_indices'], user=d['user_indices'])]
            # elif self.loss_type == 'view_enhanced_bpr':
            #     predict = [model(item=clicked['item_indices'], user=clicked['user_indices']),
            #                model(item=view['item_indices'], user=view['user_indices']),
            #                model(item=not_view['item_indices'], user=not_view['user_indices'])]
            loss = loss_function(predict)
            training_losses.append(float(loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iterations + 1) % 1000 == 0:
                validation_score = validate(model, validation_data)
                print(f'train loss: {np.array(training_losses).mean()}, '
                      f'val recall@10: {validation_score["recall"]}, '
                      f'val map@10: {validation_score["map"]}')

            if iterations > 1000 * 10:
                self.dump(model)
                break


def validate(model, data):
    user_tensor = Variable(torch.FloatTensor(data['user_index'].values)).long()
    item_tensor = Variable(torch.FloatTensor(data['item_index'].values)).long()
    scores = model(item=item_tensor, user=user_tensor)
    data['model_score'] = scores.data.numpy()
    return dict(recall=recall_at_k(data), map=map_at_k(data))


def recall_at_k(data, k=10):
    data['rank'] = data.groupby('user')['model_score'].rank(ascending=False)
    gt_clicks = data.groupby('user', as_index=False).agg({'click': 'sum'}).rename(columns={'click': 'gt_clicks'})
    gt_clicks = gt_clicks[gt_clicks['gt_clicks'] > 0]
    model_clicks = data[data['rank'] <= k].groupby('user', as_index=False).agg({'click': 'sum'}).rename(
        columns={'click': 'model_clicks'})
    clicks = pd.merge(gt_clicks, model_clicks, on='user', how='left')
    clicks['recall'] = clicks['model_clicks'] / clicks['gt_clicks']
    return clicks['recall'].mean()


def map_at_k(data, k=10):
    data['rank'] = data.groupby('user')['model_score'].rank(ascending=False)

    gt_clicks = data.groupby('user', as_index=False).agg({'click': 'sum'}).rename(columns={'click': 'gt_clicks'})
    gt_clicks['k'] = k
    gt_clicks['min'] = gt_clicks.apply(lambda x: min(x['gt_clicks'], x['k']), axis=1)

    data = data[data['rank'] <= k]
    data = data[data['click']]
    data['sum_clicks'] = data.groupby('user')['model_score'].rank(ascending=False)
    data['precision'] = data['sum_clicks'] / data['rank']
    precision_at_k = data.groupby('user', as_index=False).agg({'precision': 'sum'})

    df = pd.merge(gt_clicks, precision_at_k, on='user', how='left').fillna(0)
    df = df[df['min'] > 0]
    df['score'] = df['precision'] / df['min']
    return df['score'].mean()
