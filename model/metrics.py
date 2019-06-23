import numpy as np
from functools import lru_cache


def dcg_score(ranking, position=None):
    target = ranking if position is None else ranking[:position]
    gains = (2 ** target - 1) / np.log2(np.arange(target.size) + 2)
    return np.sum(gains)


@lru_cache()
def ndcg_score(ranking, position=None):
    ranking = np.array(ranking)
    idcg_score = dcg_score(np.sort(ranking)[::-1], position)
    return dcg_score(ranking, position) / idcg_score


def memoized_ndcg_score(ranking, position=None):
    return ndcg_score(tuple(ranking), position)


def ndcg_scores(df, query_id="listing_id", relevance="clicked"):
    return df.groupby(query_id)[relevance].agg(memoized_ndcg_score)


def mean_ndcg_score(df, query_id="listing_id", relevance="clicked"):
    return np.mean(ndcg_scores(df, query_id, relevance))
