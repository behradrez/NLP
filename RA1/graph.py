import matplotlib.pyplot as plt


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
if __name__ == "__main__":
    # Sentiment Detection - LogisticRegression
    sentiment_lr_accuracy = [0.75, 0.75, 0.75, 0.7, 0.75, 0.75, 0.8]
    sentiment_lr_precision = [0.6923076923076923, 0.6923076923076923, 0.6923076923076923, 0.6666666666666666, 0.6923076923076923, 0.6923076923076923, 0.7142857142857143]
    sentiment_lr_recall = [0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 1.0]
    
    # Sentiment Detection - LinearSVC
    sentiment_svc_accuracy = [0.7, 0.7, 0.7, 0.7, 0.7, 0.75, 0.8]
    sentiment_svc_precision = [0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6923076923076923, 0.7142857142857143]
    sentiment_svc_recall = [0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 1.0]
    
    # Plural Detection - LogisticRegression
    plural_lr_accuracy = [0.95, 0.95, 1.0, 0.95, 0.95, 1.0, 0.8]
    plural_lr_precision = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7142857142857143]
    plural_lr_recall = [0.8571428571428571, 0.8571428571428571, 1.0, 0.8571428571428571, 0.8571428571428571, 1.0, 0.7142857142857143]
    
    # Plural Detection - LinearSVC
    plural_svc_accuracy = [0.95, 0.95, 1.0, 0.9, 0.95, 1.0, 0.8]
    plural_svc_precision = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6666666666666666]
    plural_svc_recall = [0.8571428571428571, 0.8571428571428571, 1.0, 0.7142857142857143, 0.8571428571428571, 1.0, 0.8571428571428571]
    
    generate_graphs(
        sentiment_lr_accuracy, sentiment_lr_precision, sentiment_lr_recall,
        sentiment_svc_accuracy, sentiment_svc_precision, sentiment_svc_recall,
        plural_lr_accuracy, plural_lr_precision, plural_lr_recall,
        plural_svc_accuracy, plural_svc_precision, plural_svc_recall,

    )
