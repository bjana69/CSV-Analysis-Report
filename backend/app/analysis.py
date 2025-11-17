"""
Routines for parsing a CSV dataset, performing statistical analysis and
training simple machine‑learning models.  The returned dictionary contains
structures that can be consumed by the PDF generator and LLM summarisation.

The analysis deliberately avoids complex hyperparameter tuning to keep
execution fast.  Correlation matrices and descriptive statistics are
calculated using pandas and numpy.  For classification tasks, a
RandomForestClassifier is used; for regression, a RandomForestRegressor.
"""

import os
import uuid

import pandas as pd
import numpy as np
import matplotlib

# Use a non‑interactive backend so plots can be generated offscreen
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)

# Additional imports for extended analysis
from sklearn.metrics import roc_curve, auc, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency


# Helper to compute Cramer's V statistic for two categorical variables
def _cramers_v(cat1: pd.Series, cat2: pd.Series) -> float:
    """Compute Cramer's V statistic of association between two categorical variables.

    This function calculates the corrected Cramer's V value, which adjusts for bias
    in sparse contingency tables.  Returns NaN if the contingency table is not
    suitable for the calculation (e.g. one or more dimensions is 1).

    Args:
        cat1: First categorical series
        cat2: Second categorical series

    Returns:
        A float with the corrected Cramer's V statistic
    """
    confusion = pd.crosstab(cat1, cat2)
    if confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return float('nan')
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.values.sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return float('nan')
    return np.sqrt(phi2corr / denom)



def analyze_dataset(csv_path: str, task_type: str, target: str) -> dict:
    """
    Load the CSV at ``csv_path`` into a :class:`pandas.DataFrame`, perform
    extensive exploratory analysis and fit a simple machine‑learning model.
    In addition to the basic statistics returned by the previous implementation,
    this function now computes a variety of diagnostic plots and tables that
    are subsequently embedded into the generated PDF.  It also identifies
    id‑like columns and drops them from modelling, computes missing value
    statistics, outlier proportions, duplicate distributions, principal
    component projections with clustering, feature importances, and, for
    classification tasks, probabilistic calibration diagnostics and ROC curves.

    The CSV is read robustly by attempting to sniff the delimiter from the
    first few kilobytes of the file.  If sniffing fails, a comma is used as
    the default delimiter.  All columns consisting entirely of missing
    values are dropped.  Datasets with fewer than two rows are rejected.

    Args:
        csv_path: path to the CSV file on disk
        task_type: either ``"classification"`` or ``"regression"``
        target: the name of the target column (already sanitized)

    Returns:
        A dictionary containing keys used by the PDF generator.  Important
        fields include:

        ``basic_info``
            Describes the dataset shape and missing‑value ratio.

        ``missing_top5``
            A list of dictionaries describing the top five columns by missing
            percentage; each record contains column name, dtype, unique
            values, missing count and missing percent.

        ``numeric_columns``
            A list of the names of numeric columns after dropping id‑like
            columns.

        ``numeric_describe``
            Descriptive statistics for all numeric columns formatted as a
            list of records with statistics as columns.

        ``target_desc``
            Summary statistics for the target column (count, unique, top and
            frequency for categorical targets; descriptive stats for numeric).

        ``plots``
            A mapping of human‑readable labels to file paths of generated
            figures.  These include correlation heatmaps, KDE/scatter grids,
            outlier bar charts, duplicate distributions, feature importances,
            PCA clusters, target histograms, Cramer's V heatmaps, probability
            calibration scatter plots, model summary tables and ROC curves.

        ``metrics``
            Performance metrics including both train and test scores.  For
            classification tasks this includes accuracy and ROC AUC; for
            regression tasks RMSE and R².

        ``id_like_cols``
            The number of id‑like columns that were excluded from the
            modelling process.

        ``top_features``
            A list of (feature, importance) tuples sorted descending.  These
            correspond to the model after id‑like columns are removed.
    """
    # Load the dataset.  Attempt to sniff the delimiter first for
    # robustness; fall back to comma if sniffing fails.  We read the
    # entire file into memory because Pandas cannot rewind file pointers
    # after sniffing.
    import csv
    import io
    try:
        with open(csv_path, 'rb') as f:
            raw = f.read()
        sample = raw[:4096].decode('utf-8', errors='ignore')
        try:
            dialect = csv.Sniffer().sniff(sample)
            delim = dialect.delimiter
        except Exception:
            delim = ','
        df = pd.read_csv(io.BytesIO(raw), delimiter=delim)
    except Exception as exc:
        raise ValueError(f"Unable to read CSV: {exc}")
    # Drop columns that contain only missing values
    df = df.dropna(axis=1, how='all')
    # Basic validation: at least two rows
    if df.shape[0] < 2:
        raise ValueError("Dataset must contain at least two rows for analysis")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset; available columns: {list(df.columns)}")

    # Prepare features and target
    y = df[target].copy()
    # Drop the target from feature set for modelling
    X = df.drop(columns=[target]).copy()

    # Identify id-like columns.  These include columns whose name contains
    # substrings such as 'id' or 'unnamed' (case‑insensitive) as well as
    # columns with a high number of unique values relative to the number
    # of rows.  Such identifiers do not contribute to modelling and are
    # excluded from the feature set.  Record the count for reporting.
    id_like_cols: list[str] = []
    for col in X.columns:
        name = col.lower()
        # Treat columns that appear to be identifiers as id-like.  Do not
        # remove high‑cardinality measurement columns: many numeric features
        # naturally have a high proportion of unique values.  Only drop
        # columns whose names explicitly include id-like tokens such as
        # 'id' or 'unnamed'.
        if 'id' in name or 'unnamed' in name:
            id_like_cols.append(col)
    # Drop id-like columns from modelling
    if id_like_cols:
        X = X.drop(columns=id_like_cols)

    # Count of id-like columns removed
    id_like_count = len(id_like_cols)

    # Identify categorical and numeric columns for modelling.  Use X so
    # that id-like columns and the target have been removed.  These
    # variables are used in subsequent modelling, plots and metrics.  A
    # separate set of lists (summary_categorical_cols and
    # summary_numeric_cols) is computed later for reporting purposes.
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # Store high-level dataset info
    basic_info = {
        'n_rows': int(df.shape[0]),
        'n_cols': int(df.shape[1]),
        'n_numeric_cols': len(numeric_cols),
        'n_categorical_cols': len(categorical_cols),
    }
    # Compute overall missing value percentage
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isna().sum().sum()
    basic_info['missing_percent'] = float(total_missing / total_cells * 100) if total_cells > 0 else 0.0

    # Missing values per column for top 5 table
    missing_records = []
    for col in df.columns:
        miss_count = int(df[col].isna().sum())
        miss_pct = (miss_count / df.shape[0]) * 100 if df.shape[0] > 0 else 0.0
        dtype = str(df[col].dtype)
        unique_vals = int(df[col].nunique(dropna=True))
        missing_records.append({
            'column': col,
            'dtype': dtype,
            'unique': unique_vals,
            'missing_count': miss_count,
            'missing_percent': miss_pct,
        })
    # Sort by missing_count descending and take top five
    missing_top5 = sorted(missing_records, key=lambda x: x['missing_count'], reverse=True)[:5]

    # Build descriptive statistics for reporting.  Compute the list of
    # numeric columns used for the summary: all numeric columns in the
    # dataset excluding id-like columns and the target.  This ensures
    # that measurement features with a high number of unique values are
    # included in the summary instead of being treated as identifiers.
    summary_numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in id_like_cols and c != target]
    summary_categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    # Descriptive statistics for numeric columns (summary list)
    if summary_numeric_cols:
        desc_df = df[summary_numeric_cols].describe().transpose()
        desc_df['missing'] = df[summary_numeric_cols].isna().sum()
        numeric_describe = desc_df.reset_index().rename(columns={'index': 'column'}).to_dict(orient='records')
    else:
        numeric_describe = []

    # Target column description
    target_desc: dict[str, float | str] = {}
    if y.dtype == object or str(y.dtype).startswith('category'):
        # For categorical target, use value_counts
        vc = y.value_counts(dropna=False)
        top_val = vc.index[0] if not vc.empty else None
        freq = int(vc.iloc[0]) if not vc.empty else None
        target_desc = {
            'count': int(y.count()),
            'unique': int(y.nunique(dropna=True)),
            'top': str(top_val) if top_val is not None else 'N/A',
            'freq': freq if freq is not None else 0,
        }
    else:
        # For numeric target, return summary stats
        summary = y.describe()
        for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            val = summary.get(stat)
            target_desc[stat] = float(val) if pd.notnull(val) else None
        # Also include number of unique values
        target_desc['unique'] = int(y.nunique(dropna=True))
    # At this point X, y, numeric_cols, categorical_cols have been defined and id-like columns removed

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # Column transformer to one‑hot encode categorical variables while leaving
    # numeric features unchanged
    preprocessor = ColumnTransformer(
        [
            ('num', 'passthrough', numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ]
    )

    # ---------------------------------------------------------------------
    # Modelling section
    # ---------------------------------------------------------------------
    plots: dict[str, str | list[str]] = {}
    metrics: dict[str, float | None] = {}
    model = None
    # Use separate encoders for classification and regression targets
    if task_type == 'classification':
        # Encode the target if categorical
        if y.dtype == object or str(y.dtype).startswith('category'):
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
        else:
            # Ensure the target has integer labels
            y_enc = y.values
        # Check that there is more than one unique class
        unique_classes = np.unique(y_enc)
        if unique_classes.size < 2:
            raise ValueError('Classification requires at least two distinct classes in the target column')
        # Increase the number of trees to stabilise feature importances and improve alignment with
        # the reference implementation.  A higher number of estimators reduces the variance of
        # individual feature importance estimates and helps produce results similar to the
        # instructor’s report.
        model = RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        # Regression: ensure numeric target
        try:
            y_enc = y.astype(float)
        except Exception:
            raise ValueError('Target column must be numeric for regression tasks')
        # Use more estimators for regression for more stable results
        model = RandomForestRegressor(n_estimators=300, random_state=42)

    # Perform a train/test split.  Use stratify on classification problems if possible.
    try:
        if task_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=0.2, random_state=42
            )
    except Exception:
        # Fall back to unstratified split in classification if stratified fails
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42
        )

    # Fit the preprocessing and model
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    model.fit(X_train_transformed, y_train)

    # Train and test predictions for metrics
    y_pred_train = model.predict(X_train_transformed)
    y_pred_test = model.predict(X_test_transformed)

    if task_type == 'classification':
        # Classification metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        # Predicted probabilities for class 1 if available
        roc_auc_val: float | None = None
        probs_train: np.ndarray | None = None
        probs_test: np.ndarray | None = None
        try:
            probs_train = model.predict_proba(X_train_transformed)
            probs_test = model.predict_proba(X_test_transformed)
            if probs_test.shape[1] > 1:
                roc_auc_val = roc_auc_score(y_test, probs_test[:, 1])
        except Exception:
            probs_train = None
            probs_test = None
            roc_auc_val = None
        metrics = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'roc_auc': float(roc_auc_val) if roc_auc_val is not None else None,
        }
    else:
        # Regression metrics
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        metrics = {
            'train_rmse': float(rmse_train),
            'test_rmse': float(rmse_test),
            'train_r2': float(r2_train),
            'test_r2': float(r2_test),
        }

    # Compute feature importances and names on transformed columns
    def get_feature_names(transformer, num_cols, cat_cols):
        """Retrieve output feature names from a ColumnTransformer."""
        feature_names: list[str] = []
        for name, trans, cols in transformer.transformers_:
            if trans == 'passthrough':
                feature_names.extend(cols)
            else:
                try:
                    feature_names.extend(trans.get_feature_names_out(cols))
                except Exception:
                    feature_names.extend(cols)
        return feature_names

    feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(feature_names))
    idxs = np.argsort(importances)[::-1]
    top_features = [(feature_names[i], float(importances[i])) for i in idxs[:15]]

    # ------------------------------------------------------------------
    # Generate plots and diagnostic images
    # Use a unique prefix to avoid collisions across multiple API calls
    plot_prefix = uuid.uuid4().hex
    reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Correlation heatmap for numeric features (reuse existing behaviour)
    corr = df.select_dtypes(include=[np.number]).corr()
    # Use a larger figure and higher DPI to improve readability when inserted into the PDF.
    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    heatmap_filename = f"{plot_prefix}_heatmap.png"
    heatmap_path = os.path.join(reports_dir, heatmap_filename)
    plt.savefig(heatmap_path)
    plt.close()
    plots['heatmap'] = heatmap_path

    # 2. KDE or scatter plots between numeric features and target
    #    For classification tasks we follow the styling used in the provided
    #    Jupyter notebook: a single KDE curve per numeric feature with a
    #    custom colour palette.  For regression tasks we continue to use
    #    scatter plots with a fitted regression line.  The pages are
    #    arranged in a 2 column × 3 row grid (6 plots per page) and
    #    multiple pages are created as necessary.
    kde_paths: list[str] = []
    n_per_page = 6  # 2 columns × 3 rows
    if numeric_cols:
        total_plots = len(numeric_cols)
        num_pages = (total_plots + n_per_page - 1) // n_per_page
        for page_idx in range(num_pages):
            fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.69))
            fig.suptitle('KDE/Scatter Plots', fontsize=12)
            axes = axes.flatten()
            for i in range(n_per_page):
                ax = axes[i]
                col_idx = page_idx * n_per_page + i
                if col_idx >= total_plots:
                    # hide any unused subplot
                    ax.axis('off')
                    continue
                col_name = numeric_cols[col_idx]
                if task_type == 'classification':
                    # Single KDE for the entire column as per notebook styling
                    try:
                        sns.kdeplot(
                            data=df[col_name].dropna(),
                            fill=True,
                            facecolor="#4A90E2",
                            edgecolor="#003f7f",
                            alpha=0.5,
                            ax=ax
                        )
                        ax.set_title(col_name, fontsize=9)
                        ax.set_xlabel(col_name, fontsize=8)
                        ax.set_ylabel('Density', fontsize=8)
                    except Exception:
                        ax.text(0.5, 0.5, f'KDE failed for {col_name}', ha='center', va='center', fontsize=7)
                        ax.set_axis_off()
                else:
                    # Regression: scatter plot with regression line
                    try:
                        sns.regplot(
                            x=df[col_name], y=y, ax=ax,
                            scatter_kws={'s': 10, 'alpha': 0.5},
                            line_kws={'color': 'red'}
                        )
                        ax.set_title(col_name, fontsize=9)
                        ax.set_xlabel(col_name, fontsize=8)
                        ax.set_ylabel(target, fontsize=8)
                    except Exception:
                        ax.text(0.5, 0.5, f'Scatter failed for {col_name}', ha='center', va='center', fontsize=7)
                        ax.set_axis_off()
            # Adjust layout and save page
            kde_filename = f"{plot_prefix}_kde_{page_idx + 1}.png"
            kde_path = os.path.join(reports_dir, kde_filename)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(kde_path)
            plt.close(fig)
            kde_paths.append(kde_path)
    plots['kde_pages'] = kde_paths

    # 3. Histogram of target column
    try:
        plt.figure(figsize=(11, 8), dpi=150)
        if task_type == 'classification':
            # Use a count plot for categorical targets instead of a histogram.  Histplot can
            # misbehave with discrete variables on some versions of seaborn.  The count plot
            # displays the distribution of classes clearly.
            sns.countplot(data=df, x=target, palette='muted')
        else:
            sns.histplot(data=df, x=target, kde=True, color='teal')
        plt.title(f"Distribution of {target}")
        plt.ylabel('Count')
        plt.xlabel(target)
        hist_filename = f"{plot_prefix}_hist.png"
        hist_path = os.path.join(reports_dir, hist_filename)
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()
        plots['hist_target'] = hist_path
    except Exception:
        plots['hist_target'] = None

    # 4. Duplicate distribution plot: number of duplicate values per column
    dup_counts = []
    for col in df.columns:
        dup_count = df.shape[0] - df[col].nunique(dropna=False)
        dup_counts.append((col, dup_count))
    dup_counts_sorted = sorted(dup_counts, key=lambda x: x[1], reverse=True)
    try:
        # Use a larger figure and higher DPI for duplicate distribution
        plt.figure(figsize=(11, 8), dpi=150)
        cols = [c for c, _ in dup_counts_sorted]
        counts = [c for _, c in dup_counts_sorted]
        sns.barplot(y=cols, x=counts, palette='Blues_r')
        plt.title('Duplicate Distribution per Column')
        plt.xlabel('Count of Duplicate Values')
        plt.ylabel('Column')
        dup_filename = f"{plot_prefix}_duplicates.png"
        dup_path = os.path.join(reports_dir, dup_filename)
        plt.tight_layout()
        plt.savefig(dup_path)
        plt.close()
        plots['duplicates'] = dup_path
    except Exception:
        plots['duplicates'] = None

    # 5. Outlier proportion bar chart (3-sigma vs IQR)
    outlier_data = []
    for col in numeric_cols:
        series = df[col].dropna()
        # 3-sigma rule
        mean = series.mean()
        std = series.std()
        lower3 = mean - 3 * std
        upper3 = mean + 3 * std
        out3 = ((series < lower3) | (series > upper3)).sum() / max(1, len(series)) * 100
        # IQR rule
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        out_iqr = ((series < lower_iqr) | (series > upper_iqr)).sum() / max(1, len(series)) * 100
        outlier_data.append({'column': col, 'pct_3sigma': out3, 'pct_iqr': out_iqr})
    # Sort by IQR percentage
    outlier_data_sorted = sorted(outlier_data, key=lambda d: d['pct_iqr'], reverse=True)
    try:
        plt.figure(figsize=(8, 6))
        cols = [d['column'] for d in outlier_data_sorted]
        vals3 = [d['pct_3sigma'] for d in outlier_data_sorted]
        vals_iqr = [d['pct_iqr'] for d in outlier_data_sorted]
        bar_width = 0.4
        y_pos = np.arange(len(cols))
        plt.barh(y_pos - bar_width/2, vals3, height=bar_width, label='3σ', color='steelblue')
        plt.barh(y_pos + bar_width/2, vals_iqr, height=bar_width, label='IQR', color='coral')
        plt.yticks(y_pos, cols)
        plt.xlabel('% Outliers')
        plt.title('Outlier Proportion per Column (3σ vs IQR)')
        plt.legend()
        outlier_filename = f"{plot_prefix}_outliers.png"
        outlier_path = os.path.join(reports_dir, outlier_filename)
        plt.tight_layout()
        plt.savefig(outlier_path)
        plt.close()
        plots['outliers'] = outlier_path
    except Exception:
        plots['outliers'] = None

    # ------------------------------------------------------------------
    # Generate bar charts for each categorical column.  These charts
    # visualise the distribution of categories for non‑numeric columns.
    # Following the Jupyter notebook, we use the Set2 palette and
    # suppress legends.  Each chart is saved individually and the
    # collection of paths is stored in `categorical_bars`.
    # Only generate these plots if there are categorical columns.
    categorical_bar_paths: list[str] = []
    if summary_categorical_cols:
        for col in summary_categorical_cols:
            try:
                vc = df[col].value_counts(dropna=False)
                # Convert indices to strings to avoid issues with numeric or boolean types
                categories = vc.index.astype(str)
                counts = vc.values
                plt.figure(figsize=(5, 4))
                sns.barplot(x=categories, y=counts, palette='Set2')
                plt.title(f"Value counts: {col}", fontsize=10)
                plt.xlabel(col, fontsize=8)
                plt.ylabel('Count', fontsize=8)
                plt.xticks(rotation=45, ha='right', fontsize=6)
                plt.yticks(fontsize=6)
                bar_filename = f"{plot_prefix}_cat_{col}.png"
                bar_path = os.path.join(reports_dir, bar_filename)
                plt.tight_layout()
                plt.savefig(bar_path)
                plt.close()
                categorical_bar_paths.append(bar_path)
            except Exception:
                # If plotting fails for any categorical column, skip it
                continue
    # Record the categorical bar chart paths.  If none were generated the
    # list will be empty.
    plots['categorical_bars'] = categorical_bar_paths

    # 6. Cramer's V heatmap (only meaningful for classification and if categorical columns exist)
    if task_type == 'classification' and categorical_cols:
        cramers_vals = []
        for col in categorical_cols:
            val = _cramers_v(df[col], y)
            cramers_vals.append(val)
        try:
            plt.figure(figsize=(6, 4))
            cmap_vals = np.array(cramers_vals).reshape(1, -1)
            sns.heatmap(cmap_vals, annot=np.round(cmap_vals, 3), xticklabels=categorical_cols, yticklabels=[target], cmap='YlGnBu', cbar=True)
            plt.title("Cramer's V between Categorical Features and Target")
            cramers_filename = f"{plot_prefix}_cramers.png"
            cramers_path = os.path.join(reports_dir, cramers_filename)
            plt.tight_layout()
            plt.savefig(cramers_path)
            plt.close()
            plots['cramers_v'] = cramers_path
        except Exception:
            plots['cramers_v'] = None
    else:
        plots['cramers_v'] = None

    # 7. PCA reduction and KMeans clustering with silhouette score
    if numeric_cols:
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[numeric_cols].dropna())
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X_scaled)
            # Use k=2 clusters for simplicity as per requirement
            kmeans = KMeans(n_clusters=2, random_state=42)
            clusters = kmeans.fit_predict(pcs)
            # Compute silhouette score; use subset if dataset is too small
            if len(pcs) > 1:
                sil_score = silhouette_score(pcs, clusters)
            else:
                sil_score = float('nan')
            # Plot PCA scatter with clusters
            # Increase figure size and DPI for PCA cluster plots
            plt.figure(figsize=(11, 8), dpi=150)
            palette = sns.color_palette('Set2', n_colors=2)
            for cl in np.unique(clusters):
                idx = clusters == cl
                plt.scatter(pcs[idx, 0], pcs[idx, 1], s=20, label=f'Cluster {cl}', color=palette[cl])
            plt.title(f"KMeans Clustering (k=2, silhouette={sil_score:.3f})")
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend()
            pca_filename = f"{plot_prefix}_pca_clusters.png"
            pca_path = os.path.join(reports_dir, pca_filename)
            plt.tight_layout()
            plt.savefig(pca_path)
            plt.close()
            plots['pca_clusters'] = pca_path
            plots['silhouette_score'] = float(sil_score)
        except Exception:
            plots['pca_clusters'] = None
            plots['silhouette_score'] = None
    else:
        plots['pca_clusters'] = None
        plots['silhouette_score'] = None

    # 8. Feature importance bar plot
    try:
        # Use a larger figure and higher DPI for feature importance
        plt.figure(figsize=(11, 8), dpi=150)
        # Use top 15 features for readability
        top_feats = top_features[:15]
        cols = [f for f, _ in top_feats]
        vals = [v for _, v in top_feats]
        sns.barplot(x=vals, y=cols, palette='Blues_r')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f"Top Feature Importance for {target}")
        feat_filename = f"{plot_prefix}_feature_importance.png"
        feat_path = os.path.join(reports_dir, feat_filename)
        plt.tight_layout()
        plt.savefig(feat_path)
        plt.close()
        plots['feature_importance'] = feat_path
    except Exception:
        plots['feature_importance'] = None

    # 9. Predicted probability vs actual (classification) or predicted vs actual (regression)
    if task_type == 'classification' and 'roc_auc' in metrics:
        if probs_test is not None:
            try:
                # Use a larger figure and DPI for probability calibration scatter
                plt.figure(figsize=(11, 8), dpi=150)
                # Use actual class (0/1) on x axis and predicted probability on y
                sns.stripplot(x=y_test, y=probs_test[:, 1], jitter=True, alpha=0.6)
                plt.xlabel(f'Actual Class {target}')
                plt.ylabel('Predicted Probability (Class=1)')
                plt.title('Predicted Probability vs Actual Class')
                pp_filename = f"{plot_prefix}_prob_vs_actual.png"
                pp_path = os.path.join(reports_dir, pp_filename)
                plt.tight_layout()
                plt.savefig(pp_path)
                plt.close()
                plots['pred_vs_actual'] = pp_path
            except Exception:
                plots['pred_vs_actual'] = None
        else:
            plots['pred_vs_actual'] = None
    elif task_type == 'regression':
        # For regression, plot predicted vs actual scatter
        try:
            # Use a larger figure for regression predicted vs actual plots
            plt.figure(figsize=(11, 8), dpi=150)
            plt.scatter(y_test, y_pred_test, s=15, alpha=0.6)
            min_val = min(min(y_test), min(y_pred_test))
            max_val = max(max(y_test), max(y_pred_test))
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Predicted vs Actual')
            pp_filename = f"{plot_prefix}_pred_vs_actual.png"
            pp_path = os.path.join(reports_dir, pp_filename)
            plt.tight_layout()
            plt.savefig(pp_path)
            plt.close()
            plots['pred_vs_actual'] = pp_path
        except Exception:
            plots['pred_vs_actual'] = None
    else:
        plots['pred_vs_actual'] = None

    # 10. Model summary table and ROC curve (classification) or error distribution (regression)
    try:
        # Use a larger canvas for the combined model summary and ROC/residual plot
        fig, axarr = plt.subplots(1, 2, figsize=(11.69, 6.5), dpi=150)
        # Model summary table
        summary_table_data = []
        if task_type == 'classification':
            summary_table_data.append(['train_accuracy', metrics.get('train_accuracy')])
            summary_table_data.append(['test_accuracy', metrics.get('test_accuracy')])
            summary_table_data.append(['roc_auc', metrics.get('roc_auc')])
        else:
            summary_table_data.append(['train_rmse', metrics.get('train_rmse')])
            summary_table_data.append(['test_rmse', metrics.get('test_rmse')])
            summary_table_data.append(['train_r2', metrics.get('train_r2')])
            summary_table_data.append(['test_r2', metrics.get('test_r2')])
        summary_table_data.append(['#id_like_cols_not_used', id_like_count])
        summary_df = pd.DataFrame(summary_table_data, columns=['metric', 'value'])
        # Render table in the left subplot
        axarr[0].axis('off')
        table = axarr[0].table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        axarr[0].set_title('Model Summary')
        # Right subplot: ROC or residual distribution
        if task_type == 'classification' and probs_test is not None and probs_test.shape[1] > 1:
            fpr, tpr, _ = roc_curve(y_test, probs_test[:, 1])
            roc_auc_val = auc(fpr, tpr)
            axarr[1].plot(fpr, tpr, label=f'AUC = {roc_auc_val:.3f}', color='green')
            axarr[1].plot([0, 1], [0, 1], linestyle='--', color='grey')
            axarr[1].set_xlabel('FPR')
            axarr[1].set_ylabel('TPR')
            axarr[1].set_title('ROC Curve')
            axarr[1].legend(loc='lower right')
        elif task_type == 'regression':
            # Plot residuals distribution on the right
            residuals = y_pred_test - y_test
            sns.histplot(residuals, kde=True, ax=axarr[1], color='purple')
            axarr[1].set_title('Residuals Distribution')
            axarr[1].set_xlabel('Residual')
            axarr[1].set_ylabel('Count')
        else:
            axarr[1].axis('off')
        model_plot_filename = f"{plot_prefix}_model_summary.png"
        model_plot_path = os.path.join(reports_dir, model_plot_filename)
        plt.tight_layout()
        plt.savefig(model_plot_path)
        plt.close()
        plots['model_summary'] = model_plot_path
    except Exception:
        plots['model_summary'] = None


    # ------------------------------------------------------------------
    # Additional descriptive statistics for categorical variables
    # ------------------------------------------------------------------
    # Compute a simple summary (count, unique, top, freq) for each
    # categorical column.  Use summary_categorical_cols so that the
    # target column is included when it is categorical.  We record both
    # the names and statistics in the result so the PDF generator can
    # render a human‑readable table.  If there are no categorical
    # columns then categorical_describe will be an empty list.
    if summary_categorical_cols:
        cat_df = df[summary_categorical_cols]
        try:
            cat_desc_df = cat_df.describe().transpose().reset_index().rename(columns={'index': 'column'})
            categorical_describe = []
            for _, row in cat_desc_df.iterrows():
                rec = {
                    'column': str(row['column']),
                    'count': int(row['count']) if pd.notnull(row['count']) else None,
                    'unique': int(row['unique']) if pd.notnull(row['unique']) else None,
                    'top': str(row['top']) if pd.notnull(row['top']) else None,
                    'freq': int(row['freq']) if pd.notnull(row['freq']) else None,
                }
                categorical_describe.append(rec)
        except Exception:
            categorical_describe = []
    else:
        categorical_describe = []

    # After fitting, get transformed feature names from the ColumnTransformer
    def get_feature_names(transformer, num_cols, cat_cols):
        """Retrieve output feature names from a ColumnTransformer."""
        feature_names = []
        for name, trans, cols in transformer.transformers_:
            if trans == 'passthrough':
                feature_names.extend(cols)
            else:
                # OneHotEncoder provides its own feature names
                try:
                    feature_names.extend(trans.get_feature_names_out(cols))
                except Exception:
                    feature_names.extend(cols)
        return feature_names

    feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols)
    importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.zeros(len(feature_names))
    # Sort importances descending
    idxs = np.argsort(importances)[::-1]
    top_features = [(feature_names[i], float(importances[i])) for i in idxs[:10]]

    # Create a correlation heatmap for numeric features.  Use the
    # 'Reds' colour map and annotate the matrix for easier reading.  The
    # larger figure size improves clarity when embedded into an A4 page.
    corr = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='Reds', annot=True, fmt='.2f', linewidths=0.5, cbar=True)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    # Save the heatmap into the reports directory with a unique filename to
    # avoid collisions across concurrent API calls.
    heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
    heatmap_path = os.path.join(os.path.dirname(__file__), 'reports', heatmap_filename)
    plt.savefig(heatmap_path)
    plt.close()

    # Build descriptive statistics table including count, mean, std, quartiles and missing values
    desc = df.describe(include='all').transpose().reset_index().rename(columns={'index': 'column'})
    # Add missing value count
    desc['missing'] = df.isnull().sum().values
    desc_records = desc.to_dict(orient='records')

    # Compile the analysis result dictionary
    result: dict = {
        'target': target,
        'task_type': task_type,
        'metrics': metrics,
        'top_features': top_features,
        'plots': plots,
        'basic_info': basic_info,
        'missing_top5': missing_top5,
        # Use the summary lists for reporting numeric and categorical columns
        'numeric_columns': summary_numeric_cols,
        'numeric_describe': numeric_describe,
        'target_desc': target_desc,
        'categorical_columns': summary_categorical_cols,
        'categorical_describe': categorical_describe,
        'id_like_cols': id_like_count,
    }
    return result