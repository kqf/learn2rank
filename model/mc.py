import numpy as np
import pandas as pd


def generate_dataset(path, size=1000):
    categorial = {
        "c{}".format(i): np.random.randint(0, 4, size) for i in range(5)}

    numerical = {
        "n{}".format(i): np.random.random(size) for i in range(45)
    }
    additional = {
        "listing_id": np.random.randint(0, size / 10, size),
        "clicked": np.random.randint(0, 2, size),
        "doc_rank": np.arange(size)
    }
    output = {}
    output.update(numerical)
    output.update(categorial)
    output.update(additional)

    df = pd.DataFrame(output)
    df["doc_rank"] = df.groupby("listing_id")["doc_rank"].transform(
        lambda x: (x.sort_values().reset_index().index))

    df.sort_values(["listing_id", "doc_rank"], inplace=True)
    df.to_csv(path, index=False)
