"""
PDF generation using Matplotlib's PdfPages backend.

This module creates a multi‑page PDF summarising the analysis results.  Each
page is constructed as a matplotlib figure with text or images.  The
objective is not to produce a highly styled report but rather a clear and
legible document that captures key findings.  Using matplotlib avoids
external dependencies (such as reportlab or fpdf) that may not be available
in the execution environment.
"""

import os
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime
try:
    # If pytz is available, use it for timezone handling
    import pytz
except ImportError:
    pytz = None


def _add_text_page(pdf: PdfPages, title: str, lines: List[str]):
    """Helper to add a page containing a title and a list of lines."""
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size in inches
    ax.axis('off')
    y = 0.95
    # Title
    ax.text(0.5, y, title, ha='center', va='top', fontsize=16, fontweight='bold')
    y -= 0.05
    # Body text
    for line in lines:
        if y < 0.05:
            pdf.savefig(fig)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis('off')
            y = 0.95
        ax.text(0.05, y, line, ha='left', va='top', fontsize=10)
        y -= 0.03
    pdf.savefig(fig)
    plt.close(fig)


def _add_image_page(pdf: PdfPages, title: str, image_path: str):
    """Helper to add a page containing an image with a title."""
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
    ax.axis('off')
    # Title
    ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=16, fontweight='bold')
    # Insert image
    try:
        img = mpimg.imread(image_path)
        # Compute aspect ratio and adjust image size accordingly
        img_height, img_width = img.shape[0], img.shape[1]
        aspect = img_width / img_height
        # Fit image within page margins (0.05 to 0.95 horizontally and up to 0.82 vertically)
        max_width = 0.9  # fraction of page width
        max_height = 0.80  # fraction of page height (larger than before)
        # Determine width and height in figure coordinate
        if aspect > max_width / max_height:
            width = max_width
            height = width / aspect
        else:
            height = max_height
            width = height * aspect
        # Position image centred horizontally and below the title
        x0 = (1 - width) / 2
        y0 = 0.12  # start a bit above bottom, leaving room for title
        ax.imshow(img, extent=(x0, x0 + width, y0, y0 + height))
    except Exception:
        ax.text(0.5, 0.5, f"Unable to load image: {image_path}", ha='center', va='center', fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def create_pdf(report_data: Dict[str, Any], summary_text: Optional[str], pdf_path: str) -> None:
    """
    Generate a multi‑page PDF report based on the analysis results in
    ``report_data``.  The layout and styling of the pages follow the
    structure of the provided Jupyter notebook, including coloured
    headings, side‑by‑side panels for the dataset overview, heatmaps
    for descriptive statistics, and separate pages for each type of
    visualisation.  If ``summary_text`` is provided, an AI agent
    interpretation page is appended at the end.

    Args:
        report_data: Dictionary produced by ``analyze_dataset``.
        summary_text: Optional LLM summary text to include in the
            report.  If ``None`` or empty, the summary page is omitted.
        pdf_path: Destination filename for the report.  Intermediate
            directories are created as necessary.
    """
    # Create destination directory if necessary
    dest_dir = os.path.dirname(pdf_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    # Extract commonly used fields from the report
    basic = report_data.get('basic_info', {})
    missing_top5: List[Dict[str, Any]] = report_data.get('missing_top5', [])
    numeric_cols: List[str] = report_data.get('numeric_columns', [])
    numeric_desc_records: List[Dict[str, Any]] = report_data.get('numeric_describe', [])
    categorical_cols: List[str] = report_data.get('categorical_columns', [])
    categorical_desc: List[Dict[str, Any]] = report_data.get('categorical_describe', [])
    target_desc = report_data.get('target_desc', {})
    plots: Dict[str, Any] = report_data.get('plots', {})
    top_features: List[tuple] = report_data.get('top_features', [])
    metrics: Dict[str, Any] = report_data.get('metrics', {})
    task_type: str = report_data.get('task_type', '')
    target_col: str = report_data.get('target', '')
    # Determine current timestamp in Asia/Kolkata timezone
    if pytz is not None:
        tz = pytz.timezone('Asia/Kolkata')
        ts = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z')
    else:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with PdfPages(pdf_path) as pdf:
        # ------------------------------------------------------------------
        # 1. Dataset Overview page
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
        ax.axis('off')
        # Title
        ax.text(0.02, 0.96, "Dataset overview", fontsize=16, fontweight="bold", color="#1f77b4", va='top')
        # Left panel: dataset statistics
        left_lines = []
        left_lines.append(f"Generated on: {ts}")
        left_lines.append(f"Task type: {task_type}")
        left_lines.append(f"Target column: {target_col}")
        # Basic info
        if basic:
            n_rows = basic.get('n_rows', '-')
            n_cols = basic.get('n_cols', '-')
            n_num = basic.get('n_numeric_cols', '-')
            n_cat = basic.get('n_categorical_cols', '-')
            miss_pct = basic.get('missing_percent', None)
            total_missing_str = f"{miss_pct:.2f}%" if miss_pct is not None else '-' 
            left_lines.extend([
                f"Rows: {n_rows}",
                f"Columns: {n_cols}",
                f"Numeric columns: {n_num}",
                f"Categorical columns: {n_cat}",
                f"Overall missing: {total_missing_str}",
            ])
        # Model metrics (test split)
        if metrics:
            left_lines.append("")
            left_lines.append("Model Performance (test split):")
            if task_type == 'classification':
                acc = metrics.get('test_accuracy')
                auc_val = metrics.get('roc_auc')
                if acc is not None:
                    left_lines.append(f"  Accuracy: {acc:.4f}")
                if auc_val is not None:
                    left_lines.append(f"  ROC AUC: {auc_val:.4f}")
            else:
                rmse = metrics.get('test_rmse')
                r2 = metrics.get('test_r2')
                if rmse is not None:
                    left_lines.append(f"  RMSE: {rmse:.4f}")
                if r2 is not None:
                    left_lines.append(f"  R²: {r2:.4f}")
        # Top features
        if top_features:
            left_lines.append("")
            left_lines.append("Top Influencing Features:")
            for feat, imp in top_features[:5]:
                left_lines.append(f"  {feat}: {imp:.4f}")
        # ID-like columns removed
        id_like = report_data.get('id_like_cols')
        if id_like:
            left_lines.append("")
            left_lines.append(f"ID-like columns removed: {id_like}")
        # Render left text
        y_pos = 0.88
        for line in left_lines:
            ax.text(0.02, y_pos, line, fontsize=9, va='top')
            y_pos -= 0.03
        # Right panel: top missing columns
        ax.text(0.55, 0.88, "Top columns by missing values", fontsize=13, fontweight="bold", color="#1f77b4", va='top')
        y_right = 0.83
        if missing_top5:
            for rec in missing_top5:
                col = rec.get('column')
                miss_cnt = rec.get('missing_count')
                miss_pct = rec.get('missing_percent')
                line = f"{col}: {miss_cnt} ({miss_pct:.1f}%)"
                ax.text(0.55, y_right, line, fontsize=9, va='top')
                y_right -= 0.03
        else:
            ax.text(0.55, y_right, "No missing values", fontsize=9, va='top')
        pdf.savefig(fig)
        plt.close(fig)

        # ------------------------------------------------------------------
        # 2. Top 5 Missing table page
        # ------------------------------------------------------------------
        if missing_top5:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis('off')
            ax.text(0.5, 0.95, "Top 5 Columns by Missing Values", ha='center', va='top', fontsize=15, fontweight='bold', color="#1f77b4")
            # Prepare DataFrame for display
            miss_df = pd.DataFrame(missing_top5)
            # Create table with coloured header similar to notebook
            table = ax.table(cellText=miss_df.values,
                             colLabels=miss_df.columns,
                             cellLoc='center',
                             loc='center')
            # Style header
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#1f77b4")
                    cell.set_text_props(color='white', fontweight='bold')
                else:
                    cell.set_facecolor("#f7f7f7")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            pdf.savefig(fig)
            plt.close(fig)

        # ------------------------------------------------------------------
        # 3. Numeric columns list page
        # ------------------------------------------------------------------
        if numeric_cols:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis('off')
            ax.text(0.02, 0.95, "Numeric Columns", fontsize=15, fontweight='bold', color="#1f77b4", va='top')
            # Print numeric column names in a monospaced style across multiple rows
            per_row = 3
            lines = ["  |  ".join(numeric_cols[i:i+per_row]) for i in range(0, len(numeric_cols), per_row)]
            y = 0.90
            dy = 0.03
            for line in lines:
                ax.text(0.02, y, line, fontsize=9, fontfamily='monospace', va='top')
                y -= dy
            pdf.savefig(fig)
            plt.close(fig)

        # ------------------------------------------------------------------
        # 4. Numeric descriptive statistics heatmaps (chunked)
        # ------------------------------------------------------------------
        if numeric_desc_records:
            # Convert to DataFrame for easier manipulation
            desc_df = pd.DataFrame(numeric_desc_records)
            # Remove the first column if it's the index column (column name).  We assume
            # the first column is named 'column' and should remain as index rows.
            # We'll set the column names as index and then drop the 'column' column.
            if 'column' in desc_df.columns:
                desc_df = desc_df.set_index('column')
            # Determine chunk size: 5 columns per heatmap as per notebook
            chunk_size = 5
            numeric_columns_for_desc = list(desc_df.columns)
            # Each chunk corresponds to a subset of the statistics columns, not the features.
            # However, the notebook groups by numeric features; here our DataFrame is
            # features (rows) × statistics (cols).  To follow the notebook we transpose
            # the table so that features become columns and statistics become rows.
            desc_t = desc_df.T
            cols = desc_t.columns.tolist()
            for i in range(0, len(cols), chunk_size):
                chunk_cols = cols[i:i+chunk_size]
                desc_chunk = desc_t[chunk_cols]
                fig, ax = plt.subplots(figsize=(8.27, 5.5))
                cmap = sns.color_palette("mako", as_cmap=True)
                sns.heatmap(
                    desc_chunk,
                    annot=True,
                    fmt=".2g",
                    cmap=cmap,
                    linewidths=0.5,
                    linecolor="white",
                    cbar=True,
                    cbar_kws={"shrink": 0.6},
                    ax=ax
                )
                start_idx = i + 1
                end_idx = i + len(chunk_cols)
                ax.set_title(f"Numeric Summary (features {start_idx}-{end_idx})", fontsize=12)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=7)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        # ------------------------------------------------------------------
        # 5. Correlation heatmap page
        # ------------------------------------------------------------------
        heatmap_path = plots.get('heatmap')
        if heatmap_path:
            _add_image_page(pdf, "Correlation Heatmap", heatmap_path)

        # ------------------------------------------------------------------
        # 6. Categorical columns list page
        # ------------------------------------------------------------------
        if categorical_cols:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis('off')
            ax.text(0.02, 0.95, "Categorical Columns", fontsize=15, fontweight='bold', color="#1f77b4", va='top')
            per_row = 3
            lines = ["  |  ".join(categorical_cols[i:i+per_row]) for i in range(0, len(categorical_cols), per_row)]
            y = 0.90
            dy = 0.04
            for line in lines:
                ax.text(0.02, y, line, fontsize=9, fontfamily='monospace', va='top')
                y -= dy
            pdf.savefig(fig)
            plt.close(fig)

        # ------------------------------------------------------------------
        # 7. Categorical descriptive statistics table
        # ------------------------------------------------------------------
        if categorical_desc:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis('off')
            ax.text(0.5, 0.95, "Categorical Columns Summary", ha='center', va='top', fontsize=15, fontweight='bold', color="#1f77b4")
            cat_df = pd.DataFrame(categorical_desc)
            table = ax.table(cellText=cat_df.values,
                             colLabels=cat_df.columns,
                             cellLoc='center',
                             loc='center')
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#1f77b4")
                    cell.set_text_props(color='white', fontweight='bold')
                else:
                    cell.set_facecolor("#f7f7f7")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            pdf.savefig(fig)
            plt.close(fig)

        # ------------------------------------------------------------------
        # 8. Categorical value count bar charts
        # ------------------------------------------------------------------
        cat_bar_paths: List[str] = plots.get('categorical_bars', []) or []
        for bar_path in cat_bar_paths:
            # Derive a title from the filename: it contains the column name after 'cat_'
            base = os.path.basename(bar_path)
            # File pattern: {uuid}_cat_{col}.png
            title_part = base.split('_cat_')[-1].rsplit('.', 1)[0]
            title = f"Value counts: {title_part}"
            _add_image_page(pdf, title, bar_path)

        # ------------------------------------------------------------------
        # 9. Cramer's V heatmap (classification only)
        # ------------------------------------------------------------------
        cramers_path = plots.get('cramers_v')
        if cramers_path:
            _add_image_page(pdf, "Cramer's V Heatmap", cramers_path)

        # ------------------------------------------------------------------
        # 10. Histogram of the target variable
        # ------------------------------------------------------------------
        hist_path = plots.get('hist_target')
        if hist_path:
            title = f"Distribution of {target_col}"
            _add_image_page(pdf, title, hist_path)

        # ------------------------------------------------------------------
        # 11. KDE/Scatter pages
        # ------------------------------------------------------------------
        kde_pages = plots.get('kde_pages', []) or []
        for idx, kde_path in enumerate(kde_pages, start=1):
            title = f"KDE Plots – Page {idx}"
            _add_image_page(pdf, title, kde_path)

        # ------------------------------------------------------------------
        # 12. Outlier proportion page
        # ------------------------------------------------------------------
        outlier_path = plots.get('outliers')
        if outlier_path:
            _add_image_page(pdf, "Outlier Proportion per Column", outlier_path)

        # ------------------------------------------------------------------
        # 13. Duplicate distribution page
        # ------------------------------------------------------------------
        dup_path = plots.get('duplicates')
        if dup_path:
            _add_image_page(pdf, "Duplicate Distribution per Column", dup_path)

        # ------------------------------------------------------------------
        # 14. Feature importance page
        # ------------------------------------------------------------------
        feat_path = plots.get('feature_importance')
        if feat_path:
            _add_image_page(pdf, "Feature Importance", feat_path)

        # ------------------------------------------------------------------
        # 15. PCA clusters page
        # ------------------------------------------------------------------
        pca_path = plots.get('pca_clusters')
        silhouette = plots.get('silhouette_score')
        if pca_path:
            if silhouette is not None and not (isinstance(silhouette, float) and np.isnan(silhouette)):
                title = f"PCA Clusters (Silhouette = {silhouette:.3f})"
            else:
                title = "PCA Clusters"
            _add_image_page(pdf, title, pca_path)

        # ------------------------------------------------------------------
        # 16. Predicted vs Actual or Probability vs Actual
        # ------------------------------------------------------------------
        pred_path = plots.get('pred_vs_actual')
        if pred_path:
            if task_type == 'classification':
                title = "Predicted Probability vs Actual"
            else:
                title = "Predicted vs Actual"
            _add_image_page(pdf, title, pred_path)

        # ------------------------------------------------------------------
        # 17. Model summary and ROC/residuals page
        # ------------------------------------------------------------------
        model_path = plots.get('model_summary')
        if model_path:
            _add_image_page(pdf, "Model Summary & ROC/Residuals", model_path)

        # ------------------------------------------------------------------
        # 18. AI Agent interpretation (LLM summary)
        # ------------------------------------------------------------------
        if summary_text:
            summary_lines = summary_text.strip().split('\n')
            # Add a simple header
            summary_lines.insert(0, "AI Agent Interpretation")
            _add_text_page(pdf, "LLM Analysis", summary_lines)
        plt.close(fig)

        # ------------------------------------------------------------------
        # Page: Categorical columns and summary
        # ------------------------------------------------------------------
        cat_cols: List[str] = report_data.get('categorical_columns', []) or []
        cat_desc: List[Dict[str, Any]] = report_data.get('categorical_describe', []) or []
        # Only render this page if there are categorical columns or descriptive
        # statistics.  For datasets with no categorical data the page is
        # skipped entirely.
        if cat_cols or cat_desc:
            # Two‑row layout: top lists categorical columns, bottom shows the
            # summary table.  Allocate slightly more space to the bottom row.
            fig, axes = plt.subplots(
                2, 1,
                figsize=(8.27, 11.69),
                gridspec_kw={'height_ratios': [1, 2]}
            )
            # --- Top: list of categorical columns ---
            axes[0].axis('off')
            axes[0].set_title('Categorical Columns', fontsize=12)
            if cat_cols:
                # Break the list into multiple lines to avoid overflow.  Aim
                # for ~80 characters per line.  Use monospaced font for
                # alignment similar to the numeric columns list.
                lines: List[str] = []
                line = ''
                for col in cat_cols:
                    candidate = col if not line else f"{line}, {col}"
                    if len(candidate) > 80:
                        lines.append(line)
                        line = col
                    else:
                        line = candidate
                if line:
                    lines.append(line)
                y_pos = 0.9
                for l in lines:
                    axes[0].text(
                        0.02, y_pos, l,
                        fontsize=8,
                        fontfamily='monospace',
                        va='top'
                    )
                    y_pos -= 0.05
            else:
                axes[0].text(0.5, 0.5, 'No categorical columns', ha='center', va='center')
            # --- Bottom: descriptive statistics table ---
            axes[1].axis('off')
            axes[1].set_title('Categorical Summary', fontsize=12)
            if cat_desc:
                try:
                    cat_df = pd.DataFrame(cat_desc)
                    table = axes[1].table(
                        cellText=cat_df.values,
                        colLabels=cat_df.columns,
                        loc='center',
                        cellLoc='center'
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1.2, 1.5)
                except Exception:
                    axes[1].text(0.5, 0.5, 'Unable to render categorical summary', ha='center', va='center')
            else:
                axes[1].text(0.5, 0.5, 'No categorical summary available', ha='center', va='center')
            pdf.savefig(fig)
            plt.close(fig)

        # Page 3: Descriptive statistics heatmap
        numeric_describe = report_data.get('numeric_describe', [])
        if numeric_describe:
            desc_df = pd.DataFrame(numeric_describe).set_index('column')
            # Only select numeric summary columns if they exist
            stat_columns = ['count','mean','std','min','25%','50%','75%','max']
            available_stats = [c for c in stat_columns if c in desc_df.columns]
            # Break the statistics table into chunks of up to 10 columns per page
            all_columns = desc_df.index.tolist()
            if not available_stats:
                # Fallback to text representation
                stat_lines: List[str] = []
                for record in numeric_describe:
                    stat_lines.append(record.get('column', ''))
                    for key, val in record.items():
                        if key == 'column':
                            continue
                        try:
                            if pd.isna(val):
                                val_str = 'NaN'
                            elif isinstance(val, (float, int)):
                                val_str = f"{val:.4f}"
                            else:
                                val_str = str(val)
                        except Exception:
                            val_str = str(val)
                        stat_lines.append(f"  {key}: {val_str}")
                    stat_lines.append('')
                _add_text_page(pdf, 'Descriptive Statistics', stat_lines)
            else:
                # Determine chunks of up to 10 columns each
                for idx in range(0, len(all_columns), 10):
                    chunk = all_columns[idx:idx + 10]
                    heat_df = desc_df.loc[chunk, available_stats].astype(float)
                    fig, ax = plt.subplots(figsize=(8.27, 11.69))
                    sns.heatmap(heat_df, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, ax=ax)
                    page_num = idx // 10 + 1
                    ax.set_title(f'Descriptive Statistics (Numeric) – page {page_num}', fontsize=12)
                    pdf.savefig(fig)
                    plt.close(fig)
        else:
            _add_text_page(pdf,'Descriptive Statistics',['No numeric columns to describe.'])

        # Page 4: Correlation heatmap
        heatmap_path = report_data.get('plots', {}).get('heatmap')
        if heatmap_path and os.path.exists(heatmap_path):
            _add_image_page(pdf,'Correlation Heatmap',heatmap_path)

        # Page 5: Histogram and target summary
        hist_path = report_data.get('plots', {}).get('hist_target')
        target_desc = report_data.get('target_desc', {})
        # Render histogram and summary vertically (stacked) to use the full page width
        fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69), gridspec_kw={'height_ratios': [2, 1]})
        # Top: histogram
        if hist_path and os.path.exists(hist_path):
            try:
                img = mpimg.imread(hist_path)
                axes[0].imshow(img)
                axes[0].axis('off')
            except Exception:
                axes[0].text(0.5, 0.5, 'Histogram unavailable', ha='center', va='center')
        else:
            axes[0].text(0.5, 0.5, 'Histogram unavailable', ha='center', va='center')
        axes[0].set_title('Target Distribution', fontsize=12)
        # Bottom: target summary table
        axes[1].axis('off')
        if target_desc:
            td_items: List[List[Any]] = []
            for k, v in target_desc.items():
                td_items.append([k, v])
            td_df = pd.DataFrame(td_items, columns=['stat', 'value'])
            table = axes[1].table(cellText=td_df.values, colLabels=td_df.columns, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            axes[1].set_title('Target Summary', fontsize=12)
        else:
            axes[1].text(0.5, 0.5, 'No target summary available', ha='center', va='center')
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6: Cramer's V heatmap
        cramers_path = report_data.get('plots', {}).get('cramers_v')
        if cramers_path and os.path.exists(cramers_path):
            _add_image_page(pdf, "Cramer's V Heatmap", cramers_path)

        # KDE/regression scatter pages
        kde_pages = report_data.get('plots', {}).get('kde_pages', [])
        for idx, kp in enumerate(kde_pages):
            if kp and os.path.exists(kp):
                _add_image_page(pdf, f"Feature vs Target (page {idx+1})", kp)

        # Outlier bar chart
        outliers_path = report_data.get('plots', {}).get('outliers')
        if outliers_path and os.path.exists(outliers_path):
            _add_image_page(pdf,'Outlier Proportion',outliers_path)

        # Duplicate distribution chart
        duplicates_path = report_data.get('plots', {}).get('duplicates')
        if duplicates_path and os.path.exists(duplicates_path):
            _add_image_page(pdf,'Duplicate Distribution',duplicates_path)

        # Feature importance
        fi_path = report_data.get('plots', {}).get('feature_importance')
        if fi_path and os.path.exists(fi_path):
            _add_image_page(pdf,'Feature Importance',fi_path)

        # PCA clusters plot
        pca_path = report_data.get('plots', {}).get('pca_clusters')
        if pca_path and os.path.exists(pca_path):
            sil = report_data.get('plots', {}).get('silhouette_score')
            title='PCA Clusters'
            if sil is not None:
                try:
                    title=f'PCA Clusters (silhouette={sil:.3f})'
                except Exception:
                    pass
            _add_image_page(pdf,title,pca_path)

        # Predicted probability vs actual or predicted vs actual
        pp_path = report_data.get('plots', {}).get('pred_vs_actual')
        if pp_path and os.path.exists(pp_path):
            if report_data.get('task_type') == 'classification':
                _add_image_page(pdf,'Predicted Probability vs Actual',pp_path)
            else:
                _add_image_page(pdf,'Predicted vs Actual',pp_path)

        # Model summary and ROC/residuals
        model_summary_path = report_data.get('plots', {}).get('model_summary')
        if model_summary_path and os.path.exists(model_summary_path):
            _add_image_page(pdf,'Model Summary & Performance',model_summary_path)

        # LLM summary page
        if summary_text:
            summary_lines=summary_text.split('\n')
            _add_text_page(pdf,'LLM Interpretation',summary_lines)