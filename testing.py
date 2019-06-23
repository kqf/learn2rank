from scipy import stats
from model.model import build_pairwise_model, build_pointwise_model
from model.data import read_dataset
from model.metrics import mean_ndcg_score, ndcg_scores


def rank(df):
    df = df.sort_values(["listing_id", "score"], ascending=False)
    return df


train, validation = read_dataset("../data/dataset_v2.csv")
original_ndcg_scores = ndcg_scores(validation)


print(mean_ndcg_score(train))
print(mean_ndcg_score(validation))


clf = build_pointwise_model().fit(train, train["clicked"])
train["score"] = clf.predict(train)
validation["score"] = clf.predict(validation)


score = mean_ndcg_score(rank(train))
print(f"Train mean NDCG score {score}")
score = mean_ndcg_score(rank(validation))
print(f"validation mean NDCG score {score}")


groups = train.groupby("listing_id")["n0"].count().values.reshape(-1)
clf = build_pairwise_model()
clf.fit(train, train["clicked"], lgbmranker__group=groups)

train["score"] = clf.predict(train)
validation["score"] = clf.predict(validation)

score = mean_ndcg_score(rank(train))
print(f"Train mean NDCG score {score}")
score = mean_ndcg_score(rank(validation))
print(f"validation mean NDCG score {score}")


model_scores = ndcg_scores(validation)
positives = np.sum(original_ndcg_scores >= validation)
print(stats.binom_test(positives, len(model_scores)))
