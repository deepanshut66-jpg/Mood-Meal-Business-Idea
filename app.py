import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="MoodMeal Analytics Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "moodmeal_survey_synthetic_2200.csv"
MODEL_FILE = BASE_DIR / "moodmeal_model_ready_2200.csv"
DICT_FILE = BASE_DIR / "moodmeal_data_dictionary.csv"

TARGET_CLASS = "moodmeal_interest_binary"
TARGET_CLASS_LABEL = "moodmeal_interest"
TARGET_REG = "avg_order_value_numeric"
EXCLUDE_COLS = {
    "respondent_id",
    "latent_persona",
    "moodmeal_interest",
    "moodmeal_interest_3class",
    "moodmeal_interest_3class_code",
    "moodmeal_interest_ord",
    "avg_order_value_numeric",
    "monthly_food_budget_numeric",
    "comfortable_price_numeric",
}


@st.cache_data(show_spinner=False)
def load_data():
    raw_df = pd.read_csv(RAW_FILE)
    model_df = pd.read_csv(MODEL_FILE)
    dictionary_df = pd.read_csv(DICT_FILE)
    return raw_df, model_df, dictionary_df


@st.cache_data(show_spinner=False)
def get_numeric_feature_columns(model_df: pd.DataFrame):
    numeric_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in {TARGET_CLASS, TARGET_REG, "moodmeal_interest_3class_code"}]
    return features


@st.cache_resource(show_spinner=False)
def classification_results(model_df: pd.DataFrame):
    numeric_features = get_numeric_feature_columns(model_df)
    X = model_df[numeric_features].copy()
    y = model_df[TARGET_CLASS].fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])

    fi = pd.DataFrame(
        {
            "feature": numeric_features,
            "importance": clf.feature_importances_,
        }
    ).sort_values("importance", ascending=False).head(20)

    return metrics, roc_df, cm_df, fi


@st.cache_resource(show_spinner=False)
def regression_results(model_df: pd.DataFrame):
    numeric_features = get_numeric_feature_columns(model_df)
    X = model_df[numeric_features].copy()
    y = model_df[TARGET_REG].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, preds))
    metrics = {
        "R²": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": rmse,
    }

    pred_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})
    fi = pd.DataFrame(
        {"feature": numeric_features, "importance": reg.feature_importances_}
    ).sort_values("importance", ascending=False).head(20)

    return metrics, pred_df, fi


@st.cache_resource(show_spinner=False)
def clustering_results(model_df: pd.DataFrame, k: int):
    numeric_features = get_numeric_feature_columns(model_df)
    X = model_df[numeric_features].copy().fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    cluster_df = pd.DataFrame(
        {
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Cluster": labels.astype(str),
            "Order Value": model_df[TARGET_REG],
            "MoodMeal Interest": model_df[TARGET_CLASS_LABEL],
        }
    )

    summary = pd.concat(
        [
            model_df[["age_group", "occupation", "city_tier", "moodmeal_interest"]].reset_index(drop=True),
            pd.DataFrame({"Cluster": labels}),
        ],
        axis=1,
    )
    cluster_profile = summary.groupby("Cluster").agg(
        respondents=("Cluster", "size"),
        top_age_group=("age_group", lambda x: x.mode().iat[0] if not x.mode().empty else "NA"),
        top_occupation=("occupation", lambda x: x.mode().iat[0] if not x.mode().empty else "NA"),
        top_city_tier=("city_tier", lambda x: x.mode().iat[0] if not x.mode().empty else "NA"),
        top_interest=("moodmeal_interest", lambda x: x.mode().iat[0] if not x.mode().empty else "NA"),
    ).reset_index()

    return silhouette, cluster_df, cluster_profile


@st.cache_resource(show_spinner=False)
def association_results(model_df: pd.DataFrame, min_support: float, min_confidence: float):
    basket_cols = [
        c for c in model_df.columns
        if any(
            c.startswith(prefix)
            for prefix in [
                "preferred_cuisines__",
                "preferred_moodmeal_categories__",
                "preferred_meal_experience__",
                "ordering_reasons__",
                "decision_factors__",
                "personalization_preferences__",
                "personalized_meal_use_cases__",
                "trust_triggers__",
            ]
        )
    ]
    basket = model_df[basket_cols].fillna(0).astype(int)
    n = len(basket)
    supports = basket.mean(axis=0)

    rules_out = []
    for antecedent in basket_cols:
        ant_mask = basket[antecedent] == 1
        ant_support = supports[antecedent]
        if ant_support == 0:
            continue
        subset = basket.loc[ant_mask, basket_cols]
        consequent_supports = subset.mean(axis=0)
        for consequent in basket_cols:
            if antecedent == consequent:
                continue
            joint_support = (basket[antecedent] & basket[consequent]).mean()
            if joint_support < min_support:
                continue
            confidence = consequent_supports[consequent]
            if confidence < min_confidence:
                continue
            cons_support = supports[consequent]
            if cons_support == 0:
                continue
            lift = confidence / cons_support
            rules_out.append(
                {
                    "antecedents": antecedent,
                    "consequents": consequent,
                    "support": joint_support,
                    "confidence": confidence,
                    "lift": lift,
                }
            )

    rules = pd.DataFrame(rules_out)
    if rules.empty:
        return rules, basket_cols

    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    return rules, basket_cols


def render_overview(raw_df: pd.DataFrame, model_df: pd.DataFrame):
    st.title("MoodMeal Customer Analytics Dashboard")
    st.markdown("Synthetic consumer analytics app built for survey exploration, ML modeling, clustering, association mining, and regression.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", f"{len(raw_df):,}")
    c2.metric("Survey columns", raw_df.shape[1])
    c3.metric("Model-ready columns", model_df.shape[1])
    c4.metric("Interested in MoodMeal", f"{(model_df[TARGET_CLASS].mean() * 100):.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(raw_df, x="age_group", color="moodmeal_interest", barmode="group", title="Age Group vs MoodMeal Interest")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cuisine_counts = raw_df["preferred_cuisines"].fillna("Unknown").str.split(", ").explode().value_counts().reset_index()
        cuisine_counts.columns = ["Cuisine", "Count"]
        fig = px.bar(cuisine_counts.head(10), x="Cuisine", y="Count", title="Top Preferred Cuisines")
        st.plotly_chart(fig, use_container_width=True)

    spend_band = model_df["monthly_food_budget_band"].value_counts().reset_index()
    spend_band.columns = ["Budget Band", "Count"]
    fig = px.pie(spend_band, names="Budget Band", values="Count", title="Monthly Food Budget Mix")
    st.plotly_chart(fig, use_container_width=True)


def render_classification(model_df: pd.DataFrame):
    st.header("Classification: Predict MoodMeal Interest")
    st.caption("Model target: moodmeal_interest_binary")
    metrics, roc_df, cm_df, fi = classification_results(model_df)

    cols = st.columns(5)
    for idx, (name, value) in enumerate(metrics.items()):
        cols[idx].metric(name, f"{value:.3f}")

    left, right = st.columns(2)
    with left:
        roc_fig = px.line(roc_df, x="False Positive Rate", y="True Positive Rate", title="ROC Curve")
        roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1)
        st.plotly_chart(roc_fig, use_container_width=True)
    with right:
        cm_fig = px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion Matrix")
        st.plotly_chart(cm_fig, use_container_width=True)

    fi_fig = px.bar(fi.sort_values("importance"), x="importance", y="feature", orientation="h", title="Top 20 Feature Importances")
    st.plotly_chart(fi_fig, use_container_width=True)
    st.dataframe(fi, use_container_width=True)


def render_clustering(model_df: pd.DataFrame):
    st.header("Clustering: Customer Personas")
    k = st.slider("Select number of clusters (K)", min_value=2, max_value=8, value=4, step=1)
    silhouette, cluster_df, profile = clustering_results(model_df, k)
    st.metric("Silhouette Score", f"{silhouette:.3f}")

    scatter = px.scatter(
        cluster_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        title="K-Means Clusters Projected in 2D (PCA)",
        hover_data=["Order Value", "MoodMeal Interest"],
    )
    st.plotly_chart(scatter, use_container_width=True)
    st.dataframe(profile, use_container_width=True)


def render_association(model_df: pd.DataFrame):
    st.header("Association Rule Mining")
    c1, c2 = st.columns(2)
    min_support = c1.slider("Minimum support", 0.01, 0.20, 0.04, 0.01)
    min_confidence = c2.slider("Minimum confidence", 0.10, 0.95, 0.35, 0.05)

    rules, basket_cols = association_results(model_df, min_support, min_confidence)
    st.caption(f"Basket features used: {len(basket_cols)}")
    if rules.empty:
        st.warning("No rules found at the selected thresholds. Lower support or confidence and try again.")
        return

    top_rules = rules.head(25).copy()
    bubble = px.scatter(
        top_rules,
        x="support",
        y="confidence",
        size="lift",
        hover_data=["antecedents", "consequents"],
        title="Association Rules by Support, Confidence, and Lift",
    )
    st.plotly_chart(bubble, use_container_width=True)
    st.dataframe(top_rules, use_container_width=True)


def render_regression(model_df: pd.DataFrame):
    st.header("Regression: Predict Average Order Value")
    metrics, pred_df, fi = regression_results(model_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{metrics['R²']:.3f}")
    c2.metric("MAE", f"₹{metrics['MAE']:.2f}")
    c3.metric("RMSE", f"₹{metrics['RMSE']:.2f}")

    sample_pred = pred_df.sample(min(400, len(pred_df)), random_state=42)
    fig = px.scatter(sample_pred, x="Actual", y="Predicted", title="Actual vs Predicted Order Value")
    fig.add_shape(type="line", x0=sample_pred["Actual"].min(), y0=sample_pred["Actual"].min(), x1=sample_pred["Actual"].max(), y1=sample_pred["Actual"].max())
    st.plotly_chart(fig, use_container_width=True)

    fi_fig = px.bar(fi.sort_values("importance"), x="importance", y="feature", orientation="h", title="Top 20 Regression Feature Importances")
    st.plotly_chart(fi_fig, use_container_width=True)
    st.dataframe(fi, use_container_width=True)


def render_data_dictionary(dictionary_df: pd.DataFrame):
    st.header("Data Dictionary")
    st.dataframe(dictionary_df, use_container_width=True, height=600)


def main():
    raw_df, model_df, dictionary_df = load_data()

    st.sidebar.title("Navigate")
    page = st.sidebar.radio(
        "Choose a section",
        ["Overview", "Classification", "Clustering", "Association Rules", "Regression", "Data Dictionary"],
    )

    st.sidebar.markdown("---")
    st.sidebar.write("Files expected in the same folder as app.py:")
    st.sidebar.code("moodmeal_survey_synthetic_2200.csv\nmoodmeal_model_ready_2200.csv\nmoodmeal_data_dictionary.csv")

    if page == "Overview":
        render_overview(raw_df, model_df)
    elif page == "Classification":
        render_classification(model_df)
    elif page == "Clustering":
        render_clustering(model_df)
    elif page == "Association Rules":
        render_association(model_df)
    elif page == "Regression":
        render_regression(model_df)
    else:
        render_data_dictionary(dictionary_df)


if __name__ == "__main__":
    main()
