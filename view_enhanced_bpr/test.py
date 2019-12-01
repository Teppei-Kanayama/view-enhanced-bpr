import torch
from torch.autograd import Variable

import gokart

from view_enhanced_bpr.train import TrainModel
from view_enhanced_bpr.data.preprocess_data import PreprocessData
from view_enhanced_bpr.utils.evaluators import map_at_k, recall_at_k


class TestModel(gokart.TaskOnKart):
    task_namespace = 'view_enhanced_bpr'

    def requires(self):
        return dict(model=TrainModel(loss_type='view_enhanced_bpr'), data=PreprocessData())

    def run(self):
        model = self.load('model')
        data = self.load('data')['test']
        for k in range(1, 21):
            recall, map = self._run(model, data, k)
            print(k, recall, map)
        import pdb; pdb.set_trace()

    @staticmethod
    def _run(model, data, k):
        data = data[(data['click']) | (data['view'])]

        user_tensor = Variable(torch.FloatTensor(data['user_index'].values)).long()
        item_tensor = Variable(torch.FloatTensor(data['item_index'].values)).long()
        scores = model(item=item_tensor, user=user_tensor)
        data['model_score'] = scores.data.numpy()
        return recall_at_k(data, k), map_at_k(data, k)


