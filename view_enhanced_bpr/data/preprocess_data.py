import pandas as pd
import numpy as np

from logging import getLogger

import gokart
from gokart.file_processor import CsvFileProcessor
import luigi


logger = getLogger(__name__)


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
    random_state = luigi.IntParameter(default=615)  # type: int

    def requires(self):
        return LoadML100kData()

    def run(self):
        data = self.load()
        data = data.rename(columns=dict(userId='user', movieId='item')).drop('timestamp', axis=1)
        n_users = data['user'].nunique()
        n_items = data['item'].nunique()

        user_df = pd.DataFrame(dict(user=data['user'].unique(), user_index=np.arange(n_users)))
        item_df = pd.DataFrame(dict(item=data['item'].unique(), item_index=np.arange(n_items)))
        # df = _cross_join(user_df, item_df)
        # data = pd.merge(df, data, on=['user', 'item'], how='left').fillna(0.)

        data = pd.merge(data, user_df, on='user', how='left')
        data = pd.merge(data, item_df, on='item', how='left')

        data['click'] = (data['rating'] >= self.click_threshold).astype(int)
        data['view'] = (data['rating'] == 0).astype(int)
        test_data = data.sample(frac=0.2, random_state=self.random_state)
        train_data = data.drop(test_data.index)
        self.dump(dict(train=train_data, test=test_data))


def _cross_join(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df1['key'] = 0
    df2['key'] = 0
    return pd.merge(df1, df2, how='outer').drop('key', axis=1)