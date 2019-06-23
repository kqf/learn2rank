import click

from model.model import build_pairwise_model, build_pointwise_model
from model.model_selection import train_test_split_query
from model.data import read_dataset
from model.metrics import mean_ndcg_score
from model.timer import timer
from sklearn.metrics import roc_auc_score


def rank(df):
    df = df.sort_values(["listing_id", "score"], ascending=False)
    return df


@click.command()
@click.option("--path",
              type=click.Path(exists=True),
              help="Path to the dataset",
              required=True)
def pointwise(path):
    # Split the dataset, don't use validation for parameter tuning
    with timer("Split the dataset"):
        train, validation = read_dataset(path)
        train, test = train_test_split_query(train, "listing_id",
                                             test_size=0.15, random_state=137)

    with timer("Fit the model"):
        clf = build_pointwise_model().fit(train, train["clicked"])

    # Use proxy metrics for overfitting/underfitting check
    with timer("Score AUC"):
        train["score"] = clf.predict(train)
        print("Train set AUC {}".format(
            roc_auc_score(train["clicked"], train["score"]))
        )

        test["score"] = clf.predict(test)
        print("Test set AUC {}".format(
            roc_auc_score(test["clicked"], test["score"]))
        )

    with timer("Score NDCG"):
        score = mean_ndcg_score(rank(train))
        print(f"Train mean NDCG score {score}")

        score = mean_ndcg_score(rank(test))
        print(f"Test mean NDCG score {score}")


@click.command()
@click.option("--path",
              type=click.Path(exists=True),
              help="Path to the dataset",
              required=True)
def pairwise(path):
    # Split the dataset, don't use validation for parameter tuning
    with timer("Split the dataset"):
        train, validation = read_dataset(path)
        train, test = train_test_split_query(train, "listing_id",
                                             test_size=0.15, random_state=137)

    with timer("Fit the model"):
        # Count number of documents per listing
        groups = train.groupby("listing_id")["n0"].count().values.reshape(-1)
        clf = build_pairwise_model()
        clf.fit(train, train["clicked"], lgbmranker__group=groups)

    with timer("Predict the model"):
        train["score"] = clf.predict(train)
        test["score"] = clf.predict(test)

    with timer("Score NDCG"):
        score = mean_ndcg_score(rank(train))
        print(f"Train mean NDCG score {score}")

        score = mean_ndcg_score(rank(test))
        print(f"Test mean NDCG score {score}")
