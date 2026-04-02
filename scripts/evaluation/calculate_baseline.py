
import csv
import collections
import random
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # LOBSTER repo root
CSV_FILE = PROJECT_ROOT / "dataset" / "annotations" / "language_bias_annotations.jsonl"
EXCLUDED_LABELS = {'Not Enough Vote', 'Unclear / Needs Context', 'No Majority'}
VALID_LABELS = ['Negative Bias', 'Positive Bias', 'No Bias Detected']

def get_ground_truth(row):
    # row is a dict
    adj = row.get('adjudicated_decision', '').strip()
    maj = row.get('majority_decision', '').strip()
    if adj:
        return adj
    return maj

def calculate_metrics(y_true, y_pred, labels):
    # Returns macro f1
    # We need TP, FP, FN for each label
    stats = {lab: {'TP': 0, 'FP': 0, 'FN': 0} for lab in labels}
    
    for t, p in zip(y_true, y_pred):
        if t == p:
            if t in stats:
                stats[t]['TP'] += 1
        else:
            if t in stats:
                stats[t]['FN'] += 1
            if p in stats:
                stats[p]['FP'] += 1
    
    f1_scores = []
    print(f"\n{'Label':<20} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Support':<8}")
    print("-" * 65)
    
    for lab in labels:
        tp = stats[lab]['TP']
        fp = stats[lab]['FP']
        fn = stats[lab]['FN']
        support = tp + fn
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        f1_scores.append(f1)
        print(f"{lab:<20} | {prec:.4f}   | {rec:.4f}   | {f1:.4f}   | {support:<8}")
        
    macro_f1 = sum(f1_scores) / len(f1_scores)
    return macro_f1

def main():
    print(f"Loading data from {CSV_FILE}...")
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: File {CSV_FILE} not found.")
        return

    # Filter and determine ground truth
    y_true = []
    
    for row in rows:
        maj = row.get('majority_decision', '').strip()
        if maj in EXCLUDED_LABELS:
            continue
            
        gt = get_ground_truth(row)
        
        if gt == 'Unclear / Needs Context':
            continue
            
        if gt in VALID_LABELS:
            y_true.append(gt)
            
    print(f"Total valid samples: {len(y_true)}")
    
    if not y_true:
        print("No valid samples found.")
        return
        
    # Validation count
    counts = collections.Counter(y_true)
    majority_class = counts.most_common(1)[0][0]
    print(f"\nMajority Class: {majority_class} (Count: {counts[majority_class]})")
    
    # --- Majority Baseline ---
    y_pred_majority = [majority_class] * len(y_true)
    
    print("\n" + "="*50)
    print("MAJORITY BASELINE (ZeroR) RESULTS")
    print("="*50)
    macro_f1_majority = calculate_metrics(y_true, y_pred_majority, VALID_LABELS)
    print("-" * 65)
    print(f"Majority Macro F1: {macro_f1_majority:.4f}")

    # --- Stratified Random Baseline ---
    # Based on distribution
    population = list(counts.keys())
    # weights need to be in same order
    weights = [counts[k] for k in population]
    
    random.seed(42)
    y_pred_random = random.choices(population, weights=weights, k=len(y_true))
    
    print("\n" + "="*50)
    print("STRATIFIED RANDOM BASELINE RESULTS")
    print("="*50)
    macro_f1_random = calculate_metrics(y_true, y_pred_random, VALID_LABELS)
    print("-" * 65)
    print(f"Random Macro F1: {macro_f1_random:.4f}")

if __name__ == "__main__":
    main()
