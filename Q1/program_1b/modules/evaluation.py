#=====================#
# ---- libraries ---- #
#=====================#
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict, Tuple
from difflib import SequenceMatcher

#=========================#
# ---- main function ---- #
#=========================#
def calculate_text_similarity(text1: str,
                              text2: str) -> float:
    """
    Calculate similarity between two text strings.

    Parameters
    ----------
    text1 : str
        First text string
    text2 : str
        Second text string

    Returns
    -------
    float
        Similarity score between 0 and 1
    """
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def convert_to_binary_labels(predictions: List[str], ground_truth: List[str],
                             threshold: float = 0.7) -> Tuple[List[int], List[int]]:
    """
    Convert text responses to binary labels based on similarity threshold.

    Parameters
    ----------
    predictions : List[str]
        Model's predicted responses
    ground_truth : List[str]
        Ground truth responses
    threshold : float
        Similarity threshold to consider a response correct

    Returns
    -------
    Tuple[List[int], List[int]]
        Binary labels for predictions and ground truth
    """
    true_labels = []
    pred_labels = []

    for pred, true in zip(predictions, ground_truth):
        similarity = calculate_text_similarity(pred, true)

        # Convert to binary labels
        true_labels.append(1)  # Ground truth is always correct
        pred_labels.append(1 if similarity >= threshold else 0)

    return true_labels, pred_labels


def calculate_metrics(predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """
    Compute precision, recall, F1-score, and additional metrics for the agent's responses.

    Parameters
    ----------
    predictions : List[str]
        List of model's predicted responses
    ground_truth : List[str]
        List of ground truth responses

    Returns
    -------
    Dict[str, float]
        Dictionary containing various metrics
    """
    # Convert text responses to binary labels
    true_labels, pred_labels = convert_to_binary_labels(predictions, ground_truth)

    # Calculate classic metrics
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    # Calculate additional text-specific metrics
    similarities = [calculate_text_similarity(pred, true)
                    for pred, true in zip(predictions, ground_truth)]

    metrics = {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "average_similarity": round(sum(similarities) / len(similarities), 2),
        "max_similarity": round(max(similarities), 2),
        "min_similarity": round(min(similarities), 2),
        "perfect_matches": sum(s >= 0.9 for s in similarities)
    }

    return metrics


def generate_evaluation_report(queries: List[str], predictions: List[str],
                               ground_truth: List[str]) -> Tuple[Dict[str, float], str]:
    """
    Generate a comprehensive evaluation report.

    Parameters
    ----------
    queries : List[str]
        List of user queries
    predictions : List[str]
        List of model's predicted responses
    ground_truth : List[str]
        List of ground truth responses

    Returns
    -------
    Tuple[Dict[str, float], str]
        Metrics dictionary and formatted report string
    """
    metrics = calculate_metrics(predictions, ground_truth)

    report = f"""
Evaluation Report
================
Total Queries Processed: {len(queries)}

Performance Metrics
-----------------
Precision: {metrics['precision']:.2%}
Recall: {metrics['recall']:.2%}
F1 Score: {metrics['f1_score']:.2%}
Average Response Similarity: {metrics['average_similarity']:.2%}
Perfect Matches (>90% similar): {metrics['perfect_matches']}

Detailed Analysis
---------------"""

    for i, (query, pred, true) in enumerate(zip(queries, predictions, ground_truth)):
        similarity = calculate_text_similarity(pred, true)
        report += f"\n\nQuery {i + 1}: {query}"
        report += f"\nSimilarity Score: {similarity:.2%}"
        report += f"\nPredicted: {pred}"
        report += f"\nGround Truth: {true}"
        report += "\n" + "-" * 50

    return metrics, report