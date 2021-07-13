import pandas as pd

# from src.utils.config import load_config
from src.utils.store import AssignmentStore


def main():
    store = AssignmentStore()

    train_df = store.get_raw("train.csv")

    train_df = clean_input_df(train_df)
    store.put_processed("train_df_cleaned.csv", train_df)


def clean_input_df(df: pd.DataFrame) -> pd.DataFrame:
    # convert Levy from string to numeric
    df["Levy"] = df["Levy"].apply(lambda x: int(x.replace("-", "0")))

    # Convert engine volume from string to numeric and create turbo indicator
    df[["Engine volume", "turbo_ind"]] = df.apply(engine_vol, axis=1, result_type="expand")

    # Convert Mileage from string to numeric
    df["Mileage"] = pd.to_numeric(df["Mileage"].str[:-3])

    # Clean Drive wheels columns
    df["Drive wheels"] = df["Drive wheels"].replace("4x4", "All")

    # Clean doors columns
    doors_map = {"02-Mar": "two", "04-May": "four", ">5": "gt5"}
    df["Doors"] = df["Doors"].replace(doors_map)

    return df


def engine_vol(df):
    if df["Engine volume"].endswith("Turbo"):
        return float(df["Engine volume"][:-6]), "turbo"
    else:
        return float(df["Engine volume"]), "non_turbo"


if __name__ == "__main__":
    main()
