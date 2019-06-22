import click

from model.model import build_model
from model.model_selection import train_test_split_query
from model.data import read_dataset
from model.metrics import mean_ndcg_score
from model.timer import timer
from sklearn.metrics import roc_auc_score


@click.command()
@click.option("--path",
              type=click.Path(exists=True),
              help="Path to the dataset",
              required=True)
def main(path):
    # Split the dataset

    with timer("Split the dataset"):
        train, validation = read_dataset(path)
        train, test = train_test_split_query(train, "listing_id",
                                             test_size=0.15, random_state=137)

    with timer("Fit the model"):
        clf = build_model().fit(train, train["clicked"])

    with timer("Score AUC"):
        train["proba"] = clf.predict_proba(train)[:, 1]
        print("Train set AUC {}".format(
            roc_auc_score(train["clicked"], train["proba"]))
        )

        test["proba"] = clf.predict_proba(test)[:, 1]
        print("Test set AUC {}".format(
            roc_auc_score(test["clicked"], test["proba"]))
        )

    def rank(df):
        df = df.sort_values(["listing_id", "proba"], ascending=False)
        return df

    with timer("Score NDCG"):
        score = mean_ndcg_score(rank(train))
        print(f"Train mean NDCG score {score}")

        score = mean_ndcg_score(rank(test))
        print(f"Test mean NDCG score {score}")
