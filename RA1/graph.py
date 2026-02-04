import matplotlib.pyplot as plt
import csv
from pathlib import Path


def generate_graphs(
    # Sentiment Detection data
    sentiment_lr_accuracy, sentiment_lr_precision, sentiment_lr_recall,
    sentiment_svc_accuracy, sentiment_svc_precision, sentiment_svc_recall,
    # Plural Detection data
    plural_lr_accuracy, plural_lr_precision, plural_lr_recall,
    plural_svc_accuracy, plural_svc_precision, plural_svc_recall,
    x_axis=None
):
    """
    Generate two graphs for Sentiment Detection and Plural Detection.
    Saves them as separate image files.
    
    Args:
        sentiment_lr_accuracy, sentiment_lr_precision, sentiment_lr_recall: 
            Lists for LogisticRegression metrics on sentiment detection
        sentiment_svc_accuracy, sentiment_svc_precision, sentiment_svc_recall:
            Lists for LinearSVC metrics on sentiment detection
        plural_lr_accuracy, plural_lr_precision, plural_lr_recall:
            Lists for LogisticRegression metrics on plural detection
        plural_svc_accuracy, plural_svc_precision, plural_svc_recall:
            Lists for LinearSVC metrics on plural detection
        x_axis: Optional x-axis values (defaults to range if not provided)
    """
    
    # Define colors for each metric
    colors = {
        'accuracy': '#1f77b4',    # blue
        'precision': '#ff7f0e',   # orange
        'recall': '#2ca02c'       # green
    }
    
    # If x_axis not provided, create default range
    if x_axis is None:
        x_axis = list(range(len(sentiment_lr_accuracy)))
        x_axis = ['None', 'Lowercasing','Punctuation Removal','Stopword Removal','Stemming','Lemmatization','Combined']
    # ============ SENTIMENT DETECTION GRAPH ============
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # LogisticRegression - dotted lines
    ax1.plot(x_axis, sentiment_lr_accuracy, linestyle='--', color=colors['accuracy'], 
             label='LogisticRegression - Accuracy', marker='o', linewidth=2)
    ax1.plot(x_axis, sentiment_lr_precision, linestyle='--', color=colors['precision'], 
             label='LogisticRegression - Precision', marker='o', linewidth=2)
    ax1.plot(x_axis, sentiment_lr_recall, linestyle='--', color=colors['recall'], 
             label='LogisticRegression - Recall', marker='o', linewidth=2)
    
    # LinearSVC - solid lines
    ax1.plot(x_axis, sentiment_svc_accuracy, linestyle='-', color=colors['accuracy'], 
             label='LinearSVC - Accuracy', marker='s', linewidth=2)
    ax1.plot(x_axis, sentiment_svc_precision, linestyle='-', color=colors['precision'], 
             label='LinearSVC - Precision', marker='s', linewidth=2)
    ax1.plot(x_axis, sentiment_svc_recall, linestyle='-', color=colors['recall'], 
             label='LinearSVC - Recall', marker='s', linewidth=2)
    
    ax1.set_title('Sentiment Detection Performance', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Index', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_detection_performance.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # ============ PLURAL DETECTION GRAPH ============
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # LogisticRegression - dotted lines
    ax2.plot(x_axis, plural_lr_accuracy, linestyle='--', color=colors['accuracy'], 
             label='LogisticRegression - Accuracy', marker='o', linewidth=2)
    ax2.plot(x_axis, plural_lr_precision, linestyle='--', color=colors['precision'], 
             label='LogisticRegression - Precision', marker='o', linewidth=2)
    ax2.plot(x_axis, plural_lr_recall, linestyle='--', color=colors['recall'], 
             label='LogisticRegression - Recall', marker='o', linewidth=2)
    
    # LinearSVC - solid lines
    ax2.plot(x_axis, plural_svc_accuracy, linestyle='-', color=colors['accuracy'], 
             label='LinearSVC - Accuracy', marker='s', linewidth=2)
    ax2.plot(x_axis, plural_svc_precision, linestyle='-', color=colors['precision'], 
             label='LinearSVC - Precision', marker='s', linewidth=2)
    ax2.plot(x_axis, plural_svc_recall, linestyle='-', color=colors['recall'], 
             label='LinearSVC - Recall', marker='s', linewidth=2)
    
    ax2.set_title('Plural Detection Performance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Index', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plural_detection_performance.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)


# Example usage:
def _load_metrics_from_csv(csv_path):
    """Load accuracy, precision, recall lists from a CSV file.

    Expects first row to be a header and subsequent rows to have at least
    three columns: accuracy, precision, recall. Non-numeric or malformed
    rows are skipped.
    """
    acc = []
    prec = []
    rec = []
    with open(csv_path, newline='') as fh:
        reader = csv.reader(fh)
        # skip header if present
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            # try to parse first three columns as floats
            try:
                a = float(row[0])
                p = float(row[1])
                r = float(row[2])
            except (ValueError, IndexError):
                continue
            acc.append(a)
            prec.append(p)
            rec.append(r)
    return acc, prec, rec


if __name__ == "__main__":
    base = Path(__file__).parent

    # Files (expected to be in the same directory as this script)
    sentiment_lr_fp = base / 'LogisticRegression_SentimentDetectionPerformance.csv'
    sentiment_svc_fp = base / 'LinearSVC_SentimentDetectionPerformance.csv'
    plural_lr_fp = base / 'LogisticRegression_PluralDetectionPerformance.csv'
    plural_svc_fp = base / 'LinearSVC_PluralDetectionPerformance.csv'

    sentiment_lr_accuracy, sentiment_lr_precision, sentiment_lr_recall = _load_metrics_from_csv(sentiment_lr_fp)
    sentiment_svc_accuracy, sentiment_svc_precision, sentiment_svc_recall = _load_metrics_from_csv(sentiment_svc_fp)
    plural_lr_accuracy, plural_lr_precision, plural_lr_recall = _load_metrics_from_csv(plural_lr_fp)
    plural_svc_accuracy, plural_svc_precision, plural_svc_recall = _load_metrics_from_csv(plural_svc_fp)

    generate_graphs(
        sentiment_lr_accuracy, sentiment_lr_precision, sentiment_lr_recall,
        sentiment_svc_accuracy, sentiment_svc_precision, sentiment_svc_recall,
        plural_lr_accuracy, plural_lr_precision, plural_lr_recall,
        plural_svc_accuracy, plural_svc_precision, plural_svc_recall,
    )
