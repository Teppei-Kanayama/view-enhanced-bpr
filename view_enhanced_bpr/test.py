import pandas as pd
import torch
from torch.autograd import Variable

import gokart

from view_enhanced_bpr.train import TrainModel
from view_enhanced_bpr.data.preprocess_data import PreprocessData


class TestModel(gokart.TaskOnKart):
    task_namespace = 'view_enhanced_bpr'

    def requires(self):
        return dict(model=TrainModel(), data=PreprocessData())

    def run(self):
        model = self.load('model')
        data = self.load('data')['test']
        recall = self._run(model, data)
        print(recall)
        import pdb; pdb.set_trace()

    @staticmethod
    def _run(model, data):
        user_tensor = Variable(torch.FloatTensor(data['user_index'].values)).long()
        item_tensor = Variable(torch.FloatTensor(data['item_index'].values)).long()
        scores = model(item=item_tensor, user=user_tensor)
        data['model_score'] = scores.data.numpy()

        # TODO: sepalate recall@k
        # TODO: variable k
        data['rank'] = data.groupby('user')['model_score'].rank(ascending=False)

        gt_clicks = data.groupby('user', as_index=False).agg({'click': 'sum'}).rename(columns={'click': 'gt_clicks'})
        gt_clicks = gt_clicks[gt_clicks['gt_clicks'] > 0]

        model_clicks = data[data['rank'] <= 50].groupby('user', as_index=False).agg({'click': 'sum'}).rename(
            columns={'click': 'model_clicks'})
        clicks = pd.merge(gt_clicks, model_clicks, on='user', how='left')
        clicks['recall'] = clicks['model_clicks'] / clicks['gt_clicks']
        return clicks['recall'].mean()