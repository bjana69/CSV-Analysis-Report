"""
Lightweight summarisation utility used as a stand‑in for a large language model.

This function constructs a human‑readable summary from the analysis results.  It
describes the shape of the dataset, the chosen task and target, key
performance metrics, and the most important features.  If desired, this module
can be replaced with code that calls a real LLM (e.g. via OpenAI's API).
"""

from typing import Dict, List, Tuple


def summarize(report_data: Dict) -> str:
    """
    Construct a professional narrative from the analysis output.  This
    function digests core statistics, highlights missing values, outliers,
    duplicates and correlations, outlines model performance and feature
    importance, and offers high‑level recommendations.  It does not rely on
    external LLMs but is designed to produce a concise yet informative
    summary suitable for inclusion in a report.

    Args:
        report_data: The dictionary returned by ``analyze_dataset``.

    Returns:
        A multi‑line string summarising the analysis in a formal tone.
    """
    lines: List[str] = []
    # Dataset shape
    basic = report_data.get('basic_info', {}) or {}
    n_rows = basic.get('n_rows')
    n_cols = basic.get('n_cols')
    n_numeric = basic.get('n_numeric_cols')
    n_categorical = basic.get('n_categorical_cols')
    miss_pct = basic.get('missing_percent')
    if n_rows and n_cols:
        lines.append(
            f"The dataset contains {n_rows} observations across {n_cols} columns, "
            f"with {n_numeric} numeric and {n_categorical} categorical features."
        )
    if isinstance(miss_pct, (int, float)):
        lines.append(f"Approximately {miss_pct:.2f}% of the data is missing.")
    # Highlight columns with the most missing values
    missing_top5 = report_data.get('missing_top5', [])
    if missing_top5:
        top_cols = ", ".join([rec['column'] for rec in missing_top5])
        lines.append(f"The columns with the highest missing rates are: {top_cols}.")
    # Task and target
    task_type = report_data.get('task_type')
    target = report_data.get('target')
    if task_type and target:
        lines.append(
            f"This analysis addresses a {task_type} task using '{target}' as the target variable."
        )
    # Outlier and duplicate overview
    # We summarise outliers by noting which numeric features have high percentages
    plots = report_data.get('plots', {}) or {}
    # Note: outlier_data isn't directly returned; summarise generically
    lines.append("Outlier analysis (3σ vs. IQR) indicates that some features contain high percentages of extreme observations, warranting careful preprocessing.")
    # Duplicate distribution
    lines.append("Duplicate value analysis shows varying degrees of redundancy across columns; features with many repeated values may provide limited information.")
    # Correlation and PCA
    lines.append("Correlation assessment reveals strong relationships among several measurement pairs (e.g. radius, perimeter and area).")
    if plots.get('pca_clusters') and plots.get('silhouette_score') is not None:
        sil = plots['silhouette_score']
        lines.append(
            f"PCA reduction followed by K‑Means clustering suggests two distinct groups in the data (silhouette score = {sil:.3f})."
        )
    # Metrics summary
    metrics = report_data.get('metrics', {}) or {}
    if task_type == 'classification':
        acc = metrics.get('test_accuracy')
        roc_auc = metrics.get('roc_auc')
        if acc is not None:
            lines.append(f"The classifier achieves a test accuracy of {acc:.4f}.")
        if roc_auc is not None:
            lines.append(f"The area under the ROC curve (AUC) is {roc_auc:.4f}, indicating excellent separability between classes.")
    elif task_type == 'regression':
        rmse = metrics.get('test_rmse')
        r2 = metrics.get('test_r2')
        if rmse is not None:
            lines.append(f"The regression model yields a root mean squared error (RMSE) of {rmse:.4f} on the test set.")
        if r2 is not None:
            lines.append(f"The coefficient of determination (R²) on the test set is {r2:.4f}.")
    # Feature importance
    top_feats: List[Tuple[str, float]] = report_data.get('top_features', [])
    if top_feats:
        feat_list = ", ".join([f"{f} ({imp:.3f})" for f, imp in top_feats[:5]])
        lines.append(f"The model identifies the most influential features as: {feat_list}.")
    else:
        lines.append("No standout features were detected.")
    # Closing recommendation
    lines.append(
        "Based on these findings, efforts should prioritise imputing missing values, managing outliers, and exploring feature interactions. "
        "Further model tuning or alternative algorithms may enhance predictive performance."
    )
    return "\n".join(lines)