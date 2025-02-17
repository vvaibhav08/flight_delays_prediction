import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap
from sklearn.metrics import auc, roc_curve


def plot_roc_curve(y_true, y_pred_proba, title_prefix=""):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC = {roc_auc:.3f})", mode="lines", line=dict(color="blue", width=2))
    )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", mode="lines", line=dict(color="red", dash="dash")))
    fig.update_layout(
        title=f"{title_prefix}ROC Curve (AUC = {roc_auc:.3f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=700,
        height=500,
        showlegend=True,
    )

    return fig


def plot_roc_curves(y_true, model_predictions: dict, title="ROC Curves Comparison"):
    fig = go.Figure()
    colors = ["blue", "red"]

    for (model_name, y_pred_proba), color in zip(model_predictions.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, name=f"{model_name} (AUC = {roc_auc:.3f})", mode="lines", line=dict(color=color, width=2)
            )
        )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", mode="lines", line=dict(color="gray", dash="dash")))
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=1000,
        height=600,
        showlegend=True,
    )

    return fig


def plot_shap_importance(pipeline, X, feature_names=None):
    model = pipeline.steps[-1][1]
    feature_names = feature_names or X.columns.tolist()

    # Calculate SHAP values
    explainer = shap.Explainer(model, X, feature_names=feature_names)
    shap_values = explainer.shap_values(X)

    explanation = shap.Explanation(values=shap_values, data=X.values, feature_names=feature_names)

    # Create beeswarm plot
    shap.plots.beeswarm(explanation)
    return plt.gcf()
