def dataset_info(df):
    print(df.info())


def summary_statistics(df):
    return df.describe()


def missing_values(df):
    return df.isna().sum()


def duplicated_rows(df):
    return df.duplicated().sum()