import streamlit as st
import pandas as pd
import time
import tempfile

from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from joblib import Parallel, delayed


# Function to train and evaluate a model
def train_and_evaluate(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test
):
    start_time = time.time()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    runtime = time.time() - start_time

    auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)

    return model_name, auc, accuracy, runtime


# Streamlit app
st.title("Anomaly Detection Model Evaluation")

uploaded_file = st.file_uploader(
    "Choose an ARFF file",
    type="arff"
)


# Only execute after a file is uploaded
if uploaded_file is not None:

    # Save uploaded ARFF to temporary file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".arff"
    ) as tmp_file:

        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name


    # Load ARFF file
    data, meta = arff.loadarff(tmp_path)

    # Convert to DataFrame
    df = pd.DataFrame(data)


    # Dataset preview
    st.subheader("Dataset Preview")

    st.dataframe(df.head())


    # Drop ID column if it exists
    columns_to_drop = [
        col
        for col in ["id"]
        if col in df.columns
    ]

    df.drop(
        columns=columns_to_drop,
        inplace=True
    )


    # Target column
    target_column = "outlier"

    st.write(
        "Using target column:",
        target_column
    )


    # Check target exists
    if target_column not in df.columns:

        st.error(
            "The dataset does not contain an 'outlier' column."
        )

        st.stop()


    # Separate features and target
    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]


    # Decode ARFF byte labels
    if y.dtype == object:

        y = y.apply(
            lambda value:
            value.decode("utf-8")
            if isinstance(value, bytes)
            else value
        )


    # Encode target labels
    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y)


    # Display class information
    st.subheader("Dataset Information")

    st.write(
        "Target classes:",
        label_encoder.classes_
    )

    st.write("Class distribution:")

    st.dataframe(
        pd.DataFrame({
            "Class": label_encoder.classes_,
            "Count": pd.Series(y).value_counts().sort_index().values
        })
    )


    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # Standardize features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)


    # Define models
    models = {

        "KNN":
            KNeighborsClassifier(),

        "SVM":
            SVC(
                probability=True
            ),

        "Random Forest":
            RandomForestClassifier(
                random_state=42
            ),

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            )
    }


    # Run models
    with st.spinner(
        "Training models in parallel..."
    ):

        results = Parallel(
            n_jobs=-1
        )(

            delayed(
                train_and_evaluate
            )(
                name,
                model,
                X_train,
                y_train,
                X_test,
                y_test
            )

            for name, model
            in models.items()

        )


    # Convert results to DataFrame
    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "AUC",
            "Accuracy",
            "Runtime (seconds)"
        ]
    )


    # Find best model
    best_model = max(
        results,
        key=lambda x: x[1]
    )


    # Display performance
    st.subheader("Model Performance")

    st.dataframe(
        results_df,
        use_container_width=True
    )


    # Display chart
    st.subheader("AUC Comparison")

    chart_data = results_df.set_index(
        "Model"
    )[["AUC"]]

    st.bar_chart(chart_data)


    # Best model
    st.subheader("Best Model")

    st.success(
        f"""
        Best Model: {best_model[0]}

        AUC: {best_model[1]:.4f}

        Accuracy: {best_model[2]:.4f}

        Runtime: {best_model[3]:.4f} seconds
        """
    )