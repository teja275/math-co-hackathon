from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.models.regressor import SklearnRegressor
from src.utils.config import load_config
# from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore


# @validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config = load_config()

    df = store.get_processed("transformed_dataset.csv")
    df_train, df_test = train_test_split(df, test_size=config["test_size"])

    rf_estimator = RandomForestRegressor(**config["random_forest"])
    model = SklearnRegressor(rf_estimator, config["features"], config["target"])
    model.train(df_train)
    df_test["Price_predicted"] = model.predict(df_test)

    metrics = model.evaluate(df_test)

    store.put_model("saved_model.pkl", model)
    store.put_metrics("metrics.json", metrics)


if __name__ == "__main__":
    main()
