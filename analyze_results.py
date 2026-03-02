import pandas as pd
from pathlib import Path
import sys
import re

def evaluate_prediction(prediction, truth, tolerance=0.01):
    """
    Returns a dict with two metrics:
    - strict_match: The prediction is a valid number AND numerically close (validates formatting instructions).
    - semantic_match: The prediction contains the correct number somewhere, even if dirty (validates reasoning).
    """
    try:
        truth_float = float(str(truth).replace(',', '').strip())
    except ValueError:
        return {"strict": str(prediction) == str(truth), "semantic": str(prediction) == str(truth)}

    # Strict Check
    pred_str = str(prediction).strip()
    is_clean_number = False
    strict_val = None
    try:
        strict_val = float(pred_str)
        is_clean_number = True
    except ValueError:
        pass
    
    strict_hit = False
    if is_clean_number and strict_val is not None:
        if abs(truth_float) < 1e-9:
            strict_hit = abs(strict_val) < 1e-9
        else:
            diff = abs(strict_val - truth_float)
            strict_hit = (diff / abs(truth_float)) <= tolerance

    # Semantic Check 
    clean_pred_str = pred_str.lower().replace(',', '').replace('$', '').replace('€', '').replace('usd', '').replace('eur', '')
    # Extract the first valid float pattern
    match = re.search(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', clean_pred_str)
    
    semantic_hit = False
    if match:
        try:
            sem_val = float(match.group(0))
            if abs(truth_float) < 1e-9:
                semantic_hit = abs(sem_val) < 1e-9
            else:
                diff = abs(sem_val - truth_float)
                semantic_hit = (diff / abs(truth_float)) <= tolerance
        except ValueError:
            pass

    return {"strict": strict_hit, "semantic": semantic_hit}

def analyze(csv_path: str):
    path = Path(csv_path)
    if not path.exists():
        print(f"Error: Could not find {path}")
        return

    print(f"Loading {path}...")
    df = pd.read_csv(path)

    df = df.dropna(subset=['truth'])
    
    # Score each row
    eval_results = df.apply(
        lambda row: evaluate_prediction(row['message_last'], row['truth']), 
        axis=1, 
        result_type='expand'
    )
    df['strict_correct'] = eval_results['strict']
    df['semantic_correct'] = eval_results['semantic']

    # Aggregation
    stats = df.groupby('experiment').agg(
        total_runs=('eval_idx', 'count'),
        strict_acc=('strict_correct', 'mean'),
        semantic_acc=('semantic_correct', 'mean'),
        avg_latency=('time_in_seconds', 'mean'),
        avg_turns=('turns', 'mean'),
        avg_prompt_tok=('prompt_tokens', 'mean'),
        avg_compl_tok=('completion_tokens', 'mean')
    ).reset_index()

    stats['strict_acc'] = stats['strict_acc'] * 100
    stats['semantic_acc'] = stats['semantic_acc'] * 100

    # Display Results
    print("\n=== EXPERIMENT RESULTS ===")
    print(stats.to_string(float_format=lambda x: "{:.2f}".format(x)))

    # Save Summary
    summary_path = path.parent / "analysis_summary.csv"
    stats.to_csv(summary_path, index=False)
    print(f"\nSaved summary table to: {summary_path}")

    # Extract Failures
    failures = df[~df['semantic_correct']]
    if not failures.empty:
        fail_path = path.parent / "failures_semantic.csv"
        cols = ['experiment', 'eval_idx', 'eval_prompt', 'truth', 'message_last', 'strict_correct']
        # Check if special_calls exists (handling potential missing col)
        if 'special_calls' in failures.columns:
            cols.append('special_calls')
        
        failures[cols].to_csv(fail_path, index=False)
        print(f"Saved {len(failures)} semantic failures to: {fail_path}")

    # Extract Formatting Failures (Correct logic, bad format)
    format_fails = df[df['semantic_correct'] & ~df['strict_correct']]
    if not format_fails.empty:
        print(f"Found {len(format_fails)} cases where the logic was right but format was wrong (Instruction Following failure).")

if __name__ == "__main__":
    file_target = sys.argv[1] if len(sys.argv) > 1 else "logs/runs.csv"
    analyze(file_target)
