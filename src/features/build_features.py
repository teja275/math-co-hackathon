import pandas as pd
from sklearn.model_selection import train_test_split
from src.features.transformations import (
    transform_manufacturer,
    get_model_avg_price
)
from src.utils.store import AssignmentStore


def main():
    store = AssignmentStore()

    dataset = store.get_processed("train_df_cleaned.csv")
    dataset = apply_feature_engineering(dataset)

    store.put_processed("transformed_dataset.csv", dataset)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(transform_manufacturer)
        .pipe(get_model_avg_price)
    )


if __name__ == "__main__":
    main()
