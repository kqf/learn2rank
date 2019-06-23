from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler  # noqa
from sklearn.linear_model import LogisticRegression  # noqa
from lightgbm import LGBMRanker, LGBMClassifier


class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, records=False, pattern=None):
        self.columns = columns
        self.records = records

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.records:
            return X[self.columns].to_dict(orient="records")
        return X[self.columns]


class ReporteShape(BaseEstimator, TransformerMixin):
    def __init__(self, msg):
        self.msg = msg

    def fit(self, X, y):
        return self

    def transform(self, X):
        print(self.msg, X.shape)
        return X


class MeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y):
        X = X.assign(target=y)
        self.means = X.groupby(self.cols).target.mean()
        return self

    def transform(self, X):
        return X[self.cols].map(self.means).values.reshape(-1, 1)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y):
        X = X.assign(target=y)
        self.frequency = X.groupby(self.cols).size()
        return self

    def transform(self, X):
        return X[self.cols].map(self.frequency).values.reshape(-1, 1)


def categorical(colname):
    return make_pipeline(
        PandasSelector(colname, records=True),
        DictVectorizer(sparse=False),
        OneHotEncoder(categories='auto', handle_unknown='ignore'),
    )


def data_pipeline():
    # n10 is empty, the rest are the listing features
    exclude = {0, 1, 2, 10, 18, 32, 34}
    listing_features = make_union(
        FrequencyEncoder(cols="c0"),
        FrequencyEncoder(cols="c1"),
        FrequencyEncoder(cols="c2"),
        make_pipeline(
            PandasSelector(["n{}".format(i)
                            for i in range(0, 45) if i not in exclude]),
            # StandardScaler(),
        ),
    )

    document_features = make_union(
        # categorical(colname=["c3", "c4"]),
        MeanEncoder(cols="c3"),
        MeanEncoder(cols="c4"),
        make_pipeline(
            PandasSelector(["n0", "n1", "n2", "n18", "n32", "n34"]),
            # StandardScaler(),
        )
    )
    model = make_pipeline(
        make_union(
            listing_features,
            document_features,
        ),
        ReporteShape("The total shape"),
    )
    return model


def build_pointwise_model():
    model = make_pipeline(
        data_pipeline(),
        LGBMClassifier(n_estimators=200),
    )
    return model


def build_pairwise_model():
    model = make_pipeline(
        data_pipeline(),
        LGBMRanker(n_estimators=200),
    )
    return model
