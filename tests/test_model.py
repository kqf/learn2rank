from model.model import build_pointwise_model, build_pairwise_model


def test_handles_pointwise_model(data):
    train, test = data
    clf = build_pointwise_model()
    clf.fit(train, train["clicked"])
    assert len(clf.predict(train)) == len(train)
    assert len(clf.predict(test)) == len(test)


def test_handles_pairwise_model(data):
    train, test = data
    groups = train.groupby(["listing_id"])["n0"].count()
    rnk = build_pairwise_model()
    rnk.fit(train, train["clicked"], lgbmranker__group=groups)
    assert len(rnk.predict(train)) == len(train)
    assert len(rnk.predict(test)) == len(test)
