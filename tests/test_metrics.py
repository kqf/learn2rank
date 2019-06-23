import numpy as np
from model.metrics import mean_ndcg_score, memoized_ndcg_score


def test_calculates_ndcg_function(data):
    empty = (0, 0, 0, 0)
    assert np.isnan(memoized_ndcg_score(empty))

    semioptimal = (0, 1, 0, 0)
    assert 0 < memoized_ndcg_score(semioptimal) < 1.

    optimal1 = (1, 0, 0, 0)
    assert memoized_ndcg_score(optimal1) == 1.

    optimal2 = (1, 1, 1, 1)
    assert memoized_ndcg_score(optimal2) == 1.


def test_calculates_mean_scores(data):
    train, test = data
    assert 0 < mean_ndcg_score(train) < 1.
    assert 0 < mean_ndcg_score(test) < 1.
