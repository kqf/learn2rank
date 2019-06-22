import os

import pandas as pd
import numpy as np
import joblib
from model.model_selection import train_test_split_query


DIR_ROOT = os.path.dirname(os.path.realpath(__file__))
CACHEDIR = os.path.join(DIR_ROOT, '../data/.cachedir/')
memory = joblib.Memory(cachedir=CACHEDIR, verbose=0)


@memory.cache()
def read_dataset(path, test_size=0.15):
    df = pd.read_csv(path)
    #  Cleanup: remove all the listings without clicks in
    # as the first approximation. In fact this information is useful as well,
    # but there's no labels so we can't evaluate our performance on that data.

    labeled = df.groupby("listing_id")["clicked"].transform(np.sum) > 0
    df = df[labeled]
    return train_test_split_query(df, "listing_id",
                                  test_size=test_size, random_state=137)
