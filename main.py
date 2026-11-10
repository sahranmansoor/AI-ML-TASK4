import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
        confusion_matrix,
        ConfusionMatrixDisplay,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve
    )


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

    cm = confusion_matrix(y_test, y_pred)

    print(cm)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"]).plot()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    precision = precision_score(y_test, y_pred)
    print("precision: ", precision)

    recall = recall_score(y_test, y_pred)
    print("recall: ", recall)

    y_prob = model.predict_proba(x_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print("ROC_AUC: ", auc)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
