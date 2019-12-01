import numpy as np
import torch
from torch.autograd import Variable
import gokart
import luigi

from view_enhanced_bpr.data.preprocess_data import PreprocessData, MakeTrainPair, MakeTrainTriplet
from view_enhanced_bpr.model.matrix_factorization import MatrixFactorization
from view_enhanced_bpr.model.loss import bpr_loss, view_enhanced_bpr_loss
from view_enhanced_bpr.utils.evaluators import map_at_k, recall_at_k


class TrainModel(gokart.TaskOnKart):
    task_namespace = 'view_enhanced_bpr'
    embedding_dim = luigi.IntParameter(default=10)  # type: int
    lr = luigi.FloatParameter(default=0.005)  # type: float
    weight_decay = luigi.FloatParameter(default=0.0001)  # type: float
    alpha = luigi.FloatParameter(default=0.5)  # type: float
    loss_type = luigi.Parameter(default='view_enhanced_bpr')  # type: str

    def requires(self):
        return dict(data=PreprocessData(), train_pair_data=MakeTrainPair(), train_triplet_data=MakeTrainTriplet())

    def output(self):
        return self.make_model_target(relative_file_path='model/mf.zip',
                                      save_function=torch.save,
                                      load_function=torch.load)

    def run(self):
        data = self.load('data')['train']
        n_users = data['user_index'].max() + 1
        n_items = data['item_index'].max() + 1

        train_pair_data = self.load('train_pair_data')
        train_triplet_data = self.load('train_triplet_data')
        validation_data = self.load('data')['validation']

        model = MatrixFactorization(n_items=n_items, n_users=n_users, embedding_dim=self.embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        loss_functions = dict(
            bpr=bpr_loss,
            view_enhanced_bpr=view_enhanced_bpr_loss
            )
        loss_function = loss_functions[self.loss_type]

        if self.loss_type == 'bpr':
            data = model.data_sampler(train_pair_data)
        elif self.loss_type == 'view_enhanced_bpr':
            data = model.data_sampler2(train_triplet_data)

        training_losses = []
        for iterations, d in enumerate(data):
            if self.loss_type == 'bpr':
                predict = [model(item=d['clicked_item_indices'], user=d['user_indices']),
                           model(item=d['not_clicked_item_indices'], user=d['user_indices'])]
            elif self.loss_type == 'view_enhanced_bpr':
                predict = [model(item=d['clicked_item_indices'], user=d['user_indices']),
                           model(item=d['viewed_item_indices'], user=d['user_indices']),
                           model(item=d['not_viewed_item_indices'], user=d['user_indices'])]
            loss = loss_function(predict)
            training_losses.append(float(loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iterations % 1000 == 0:
                validation_score = validate(model, validation_data)
                print(f'train loss: {np.array(training_losses).mean()}, '
                      f'val recall@10: {validation_score["recall"]}, '
                      f'val map@10: {validation_score["map"]}')

            if iterations > 1000 * 10:
                self.dump(model)
                break


def validate(model, data):
    # data = data[~data['view']]
    user_tensor = Variable(torch.FloatTensor(data['user_index'].values)).long()
    item_tensor = Variable(torch.FloatTensor(data['item_index'].values)).long()
    scores = model(item=item_tensor, user=user_tensor)
    data['model_score'] = scores.data.numpy()
    return dict(recall=recall_at_k(data, k=10), map=map_at_k(data, k=10))
