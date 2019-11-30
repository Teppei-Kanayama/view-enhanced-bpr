import pandas as pd
import numpy as np
import gokart
from gokart.file_processor import CsvFileProcessor
import luigi


def _cross_join(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df1['key'] = 0
    df2['key'] = 0
    return pd.merge(df1, df2, how='outer').drop('key', axis=1)


class _ML100kDataProcessor(CsvFileProcessor):
    def load(self, file):
        try:
            return pd.read_csv(file, sep='\t', header=None).rename(columns={0: 'userId', 1: 'movieId', 2: 'rating', 3: 'timestamp'})
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


class LoadML100kData(gokart.TaskOnKart):
    task_namespace = 'view_enhanced_bpr'

    def output(self):
        processor = _ML100kDataProcessor()
        return self.make_target('ml-100k/ml-100k.tsv', use_unique_id=False, processor=processor)


class PreprocessData(gokart.TaskOnKart):
    task_namespace = 'view_enhanced_bpr'

    click_threshold = luigi.IntParameter(default=4)  # type: int
    test_ratio = luigi.FloatParameter(default=0.3)  # type: float
    validation_ratio = luigi.FloatParameter(default=0.3)  # type: float

    def requires(self):
        return LoadML100kData()

    def run(self):
        data = self.load()
        data = data.rename(columns=dict(userId='user', movieId='item')).drop('timestamp', axis=1)
        n_users = data['user'].nunique()
        n_items = data['item'].nunique()

        user_df = pd.DataFrame(dict(user=data['user'].unique(), user_index=np.arange(n_users)))
        item_df = pd.DataFrame(dict(item=data['item'].unique(), item_index=np.arange(n_items)))
        df = _cross_join(user_df, item_df)
        data = pd.merge(df, data, on=['user', 'item'], how='left').fillna(0.)

        data['click'] = data['rating'] >= self.click_threshold
        data['view'] = data['rating'] == 0

        test_data = data[(data['user_index'] < n_users * self.test_ratio) & (data['item_index'] < n_items * self.test_ratio)]
        validation_data = data[(data['user_index'] > n_users * (1 - self.validation_ratio)) & (
                    data['item_index'] > n_items * (1 - self.validation_ratio))]

        train_data = data.drop(list(validation_data.index) + list(test_data.index))
        self.dump(dict(train=train_data, validation=validation_data, test=test_data))


class MakeTrainPair(gokart.TaskOnKart):
    task_namespace = 'view_enhanced_bpr'

    positive_sample_weight = luigi.IntParameter(default=5)

    def requires(self):
        return PreprocessData()

    def run(self):
        data = self.load()['train']

        clicked_data = data[data['click']].rename(columns={'item_index': 'clicked_item_index'})
        not_clicked_data = data[~data['click']].rename(columns={'item_index': 'not_clicked_item_index'})
        not_clicked_data = not_clicked_data.groupby('user_index').apply(lambda x: x.sample(self.positive_sample_weight)).reset_index(drop=True)

        paired_data = pd.merge(clicked_data[['user_index', 'clicked_item_index']],
                               not_clicked_data[['user_index', 'not_clicked_item_index']],
                               on='user_index', how='inner')
        self.dump(paired_data)


class MakeTrainTriplet(gokart.TaskOnKart):
    task_namespace = 'view_enhanced_bpr'

    positive_sample_weight = luigi.IntParameter(default=5)

    def requires(self):
        return PreprocessData()

    def run(self):
        data = self.load()['train']

        clicked_data = data[data['click']].rename(columns={'item_index': 'clicked_item_index'})
        viewed_data = data[data['view']].rename(columns={'item_index': 'viewed_item_index'})
        not_viewed_data = data[(~data['click']) & (~data['view'])].rename(columns={'item_index': 'not_viewed_item_index'})

        viewed_data = viewed_data.groupby('user_index').apply(lambda x: x.sample(self.positive_sample_weight)).reset_index(drop=True)

        triplet_data = pd.merge(clicked_data[['user_index', 'clicked_item_index']],
                                viewed_data[['user_index', 'viewed_item_index']],
                                on='user_index', how='inner')
        triplet_data = pd.merge(triplet_data,
                                not_viewed_data[['user_index', 'not_viewed_item_index']],
                                on='user_index', how='inner')
        self.dump(triplet_data)
