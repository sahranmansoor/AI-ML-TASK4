import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    df = pd.read_csv("data.csv")

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df["diagnosis"] = df["diagnosis"].map({ "M": 1, "B": 0})

    X = df.drop("diagnosis", axis=1)
    Y = df["diagnosis"]

    x_train, x_test, y_train, y_test = train_test_split(
            X, Y,
            test_size=0.2,
            random_state=1
            )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

if __name__ == "__main__":
    main()
