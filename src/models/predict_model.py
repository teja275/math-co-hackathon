import numpy as np
import pandas as pd

from src.features.build_features import apply_feature_engineering
# from src.utils.guardrails import validate_prediction_results
from src.utils.store import AssignmentStore


# @validate_prediction_results
def main():
    store = AssignmentStore()

    df_test = store.get_raw("test.csv")
    df_test = apply_feature_engineering(df_test)

    model = store.get_model("saved_model.pkl")
    df_test["Price"] = model.predict(df_test)

    store.put_predictions("results.csv", df_test[["Price"]])


if __name__ == "__main__":
    main()
