import streamlit as st
import pandas as pd
import time
import tempfile
from pathlib import Path

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from joblib import Parallel, delayed


st.set_page_config(
    page_title="Anomaly Detection Model Evaluation",
    page_icon="📊",
    layout="centered",
)

SAMPLE_DIR = Path(__file__).parent / "sample_data"

SAMPLE_DATASETS = {
    "Annthyroid": SAMPLE_DIR / "Annthyroid.arff",
    "Heart Disease": SAMPLE_DIR / "Heart_Disease.arff",
    "Pima": SAMPLE_DIR / "Pima.arff",
    "WPBC": SAMPLE_DIR / "WPBC.arff",
}


def train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    runtime = time.time() - start_time

    return (
        model_name,
        roc_auc_score(y_test, y_score),
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred, zero_division=0),
        runtime,
    )


def load_arff(file_source):
    data, _ = arff.loadarff(file_source)
    return pd.DataFrame(data)


st.title("Anomaly Detection Model Evaluation")

st.write(
    """
    Compare machine learning models for anomaly detection on imbalanced
    healthcare datasets. Select a sample dataset or upload your own ARFF file
    to benchmark models across AUC, accuracy, precision, recall, F1 score,
    and runtime.
    """
)

dataset_choice = st.selectbox(
    "Choose a dataset",
    [
        "Annthyroid",
        "Heart Disease",
        "Pima",
        "WPBC",
        "Upload my own ARFF file",
    ],
)

file_source = None
dataset_name = None

if dataset_choice == "Upload my own ARFF file":
    uploaded_file = st.file_uploader(
        "Upload an ARFF dataset",
        type="arff",
        help="The dataset must contain a binary target column named 'outlier'.",
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".arff") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_source = tmp_file.name
            dataset_name = uploaded_file.name
else:
    file_source = SAMPLE_DATASETS[dataset_choice]
    dataset_name = dataset_choice

if file_source is not None:
    try:
        df = load_arff(file_source)
    except Exception as error:
        st.error(f"Unable to load the ARFF file: {error}")
        st.stop()

    st.caption(f"Evaluating: {dataset_name}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    columns_to_drop = [col for col in ["id"] if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)

    target_column = "outlier"

    if target_column not in df.columns:
        st.error("The dataset does not contain an 'outlier' column.")
        st.stop()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if y.dtype == object:
        y = y.apply(
            lambda value: value.decode("utf-8")
            if isinstance(value, bytes)
            else value
        )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    if len(label_encoder.classes_) != 2:
        st.error(
            "The 'outlier' column must contain exactly two classes "
            "for binary anomaly detection."
        )
        st.stop()

    if X.isnull().any().any():
        st.error(
            "The dataset contains missing feature values. "
            "Please preprocess missing values before evaluation."
        )
        st.stop()

    st.subheader("Dataset Overview")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Records", f"{len(df):,}")
    metric_col2.metric("Features", X.shape[1])
    metric_col3.metric("Anomaly Rate", f"{y.mean() * 100:.2f}%")

    class_distribution = pd.DataFrame(
        {
            "Class": label_encoder.classes_,
            "Count": pd.Series(y).value_counts().sort_index().values,
        }
    )

    st.write("**Target column:** `outlier`")
    st.write("**Class distribution:**")
    st.dataframe(class_distribution, use_container_width=True, hide_index=True)

    st.info(
        """
        This dataset is imbalanced. Accuracy alone may be misleading because
        a model can achieve high accuracy by primarily predicting the majority
        class. AUC is therefore used as the primary model-selection metric,
        while precision, recall, and F1 provide additional context.
        """
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    with st.spinner("Training and evaluating models in parallel..."):
        results = Parallel(n_jobs=-1)(
            delayed(train_and_evaluate)(
                name, model, X_train, y_train, X_test, y_test
            )
            for name, model in models.items()
        )

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "AUC",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Runtime (seconds)",
        ],
    )

    best_model = max(results, key=lambda result: result[1])

    st.subheader("Model Performance")

    display_df = results_df.copy()
    numeric_columns = [
        "AUC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Runtime (seconds)",
    ]
    display_df[numeric_columns] = display_df[numeric_columns].round(4)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("AUC Comparison")
    chart_data = results_df.sort_values("AUC").set_index("Model")[["AUC"]]
    st.bar_chart(chart_data)

    st.subheader("Best Model by AUC")

    st.success(
        f"""
        **{best_model[0]}** achieved the highest AUC.

        **AUC:** {best_model[1]:.4f}

        **Accuracy:** {best_model[2]:.4f}

        **Precision:** {best_model[3]:.4f}

        **Recall:** {best_model[4]:.4f}

        **F1 Score:** {best_model[5]:.4f}

        **Runtime:** {best_model[6]:.4f} seconds
        """
    )
