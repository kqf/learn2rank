from sklearn.model_selection import train_test_split


def train_test_split_query(df, query_col, **options):
    queries = df[query_col].unique()
    train_queries, test_queries = train_test_split(queries, **options)

    def drop(queries):
        return df.loc[df[query_col].isin(set(queries))]

    return drop(train_queries), drop(test_queries)
