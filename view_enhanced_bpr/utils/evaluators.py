import pandas as pd


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
