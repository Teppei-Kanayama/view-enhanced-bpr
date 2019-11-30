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

