import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.nn import LogSigmoid
import gokart
import luigi

from view_enhanced_bpr.data.preprocess_data import PreprocessData
from view_enhanced_bpr.model.matrix_factorization import MatrixFactorization


class TrainModel(gokart.TaskOnKart):
    task_namespace = 'view_enhanced_bpr'
    validation_ratio = luigi.FloatParameter(default=0.1)  # type: float
    embedding_dim = luigi.IntParameter(default=10)  # type: int
    lr = luigi.FloatParameter(default=0.0001)  # type: float
    weight_decay = luigi.FloatParameter(default=0.0)  # type: float
    alpha = luigi.FloatParameter(default=0.5)  # type: float

    def requires(self):
        return PreprocessData()

    def output(self):
        return self.make_model_target(relative_file_path='model/mf.zip',
                                      save_function=torch.save,
                                      load_function=torch.load)

    def run(self):
        data = self.load()['train']
        n_users = data['user_index'].max() + 1
        n_items = data['item_index'].max() + 1

        validation_data = data[
            (data['user_index'] > n_users * (1 - self.validation_ratio)) & (data['item_index'] > n_items * (1 - self.validation_ratio))]
        train_data = data.drop(validation_data.index)

        model = MatrixFactorization(n_items=n_items, n_users=n_users, embedding_dim=self.embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        clicked_data = model.data_sampler(train_data, key='click')
        not_click_data = model.data_sampler(train_data, key='not_click')
        view_data = model.data_sampler(train_data, key='view')
        not_view_data = model.data_sampler(train_data, key='not_view')

        training_losses = []
        for iterations, (clicked, not_clicked, view, not_view) in enumerate(zip(clicked_data, not_click_data, view_data, not_view_data)):
            # TODO: refactor
            predict1 = model(item=clicked['item_indices'], user=clicked['user_indices'])
            predict2 = model(item=not_view['item_indices'], user=not_view['user_indices'])
            predict3 = model(item=view['item_indices'], user=view['user_indices'])

            # TODO: define loss function
            # loss = -LogSigmoid()(predict1 - predict2).mean()
            loss = (- LogSigmoid()(predict1 - predict2)
                    - self.alpha * LogSigmoid()(predict1 - predict3)
                    - (1 - self.alpha) * LogSigmoid()(predict3 - predict2)).mean()

            training_losses.append(float(loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iterations + 1) % 1000 == 0:
                print(f'train loss: {np.array(training_losses).mean()}, val recall: {validate(model, validation_data)}')

            if iterations > 3000000:
                self.dump(model)
                break


class TestModel(gokart.TaskOnKart):
    task_namespace = 'view_enhanced_bpr'

    def requires(self):
        return TrainModel()

    def run(self):
        model = self.load()
        import pdb; pdb.set_trace()


def validate(model, data):
    user_tensor = Variable(torch.FloatTensor(data['user_index'].values)).long()
    item_tensor = Variable(torch.FloatTensor(data['item_index'].values)).long()
    scores = model(item=item_tensor, user=user_tensor)
    data['model_score'] = scores.data.numpy()

    # TODO: sepalate recall@k
    # TODO: variable k
    data['rank'] = data.groupby('user')['model_score'].rank(ascending=False)

    gt_clicks = data.groupby('user', as_index=False).agg({'click': 'sum'}).rename(columns={'click': 'gt_clicks'})
    gt_clicks = gt_clicks[gt_clicks['gt_clicks'] > 0]

    model_clicks = data[data['rank'] <= 50].groupby('user', as_index=False).agg({'click': 'sum'}).rename(columns={'click': 'model_clicks'})
    clicks = pd.merge(gt_clicks, model_clicks, on='user', how='left')
    clicks['recall'] = clicks['model_clicks'] / clicks['gt_clicks']
    return clicks['recall'].mean()
