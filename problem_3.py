"""
Biostatistics Homework 3 - Problem 3
Author: Logan Kaising
Date: March 2026
Software: Python 3.11.13 with numpy, scipy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "hand_span_data.csv"
ALPHA = 0.05
FIG_DIR = "figures"


# ============================================================
# DATA LOADING & CLEANING
# ============================================================

def load_and_clean(csv_path):
    """
    Load the hand span dataset and return a cleaned DataFrame
    with numeric columns and stripped string columns.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    for col in ["Right Hand Span (cm)", "Left Hand Span (cm)", "Height (cm)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["Biological Sex", "Sports", "Music"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df


# ============================================================
# TEST FUNCTIONS
# ============================================================

def get_t_critical(alpha, df, tail='two'):
    """
    Get the critical t-value for the given alpha, df, and tail direction.
    """
    if tail == 'two':
        return stats.t.ppf(1 - alpha / 2, df)
    elif tail == 'right':
        return stats.t.ppf(1 - alpha, df)
    elif tail == 'left':
        return stats.t.ppf(alpha, df)
    else:
        raise ValueError(f"Invalid tail type: {tail}")


def calculate_p_value_t(t, df, tail='two'):
    """
    Calculate the p-value for the given t-score and tail direction.
    """
    if tail == 'two':
        return 2 * (1 - stats.t.cdf(abs(t), df))
    elif tail == 'right':
        return 1 - stats.t.cdf(t, df)
    elif tail == 'left':
        return stats.t.cdf(t, df)
    else:
        raise ValueError(f"Invalid tail type: {tail}")


# ============================================================
# PART (a): TWO-SAMPLE POOLED T-TEST — MALE vs FEMALE HAND SPAN
# ============================================================

def run_part_a(df):
    """
    Two-sample pooled t-test comparing male vs female average hand span.
    Average = (right + left) / 2 per individual, then compare by sex.
    """
    df_a = df.copy()
    df_a["avg_span"] = (df_a["Right Hand Span (cm)"] + df_a["Left Hand Span (cm)"]) / 2

    male = df_a.loc[df_a["Biological Sex"] == "Male", "avg_span"].dropna().to_numpy()
    female = df_a.loc[df_a["Biological Sex"] == "Female", "avg_span"].dropna().to_numpy()

    n1, n2 = len(male), len(female)
    x1_bar, x2_bar = np.mean(male), np.mean(female)
    s1, s2 = np.std(male, ddof=1), np.std(female, ddof=1)

    df_val = n1 + n2 - 2
    s_pooled_sq = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df_val
    t_stat = (x1_bar - x2_bar) / np.sqrt(s_pooled_sq * (1/n1 + 1/n2))

    t_crit = get_t_critical(ALPHA, df_val, tail='two')
    p_value = calculate_p_value_t(t_stat, df_val, tail='two')
    reject_null = abs(t_stat) > t_crit

    t_scipy, p_scipy = stats.ttest_ind(male, female, equal_var=True)

    results = {
        'n1': n1, 'n2': n2,
        'x1_bar': x1_bar, 'x2_bar': x2_bar,
        's1': s1, 's2': s2,
        's_pooled_sq': s_pooled_sq,
        's_pooled': np.sqrt(s_pooled_sq),
        'df': df_val,
        't_stat': t_stat, 't_crit': t_crit,
        'p_value': p_value, 'reject_null': reject_null,
        't_scipy': t_scipy, 'p_scipy': p_scipy,
    }

    return results


# ============================================================
# PART (b): PAIRED T-TEST — FEMALE RIGHT vs LEFT HAND SPAN
# ============================================================

def run_part_b(df):
    """
    Paired t-test comparing female right-hand span vs left-hand span.
    """
    fem = df[df["Biological Sex"] == "Female"].copy()
    rhs = fem["Right Hand Span (cm)"]
    lhs = fem["Left Hand Span (cm)"]
    mask = rhs.notna() & lhs.notna()
    rhs = rhs[mask].to_numpy()
    lhs = lhs[mask].to_numpy()

    d = rhs - lhs
    n = len(d)
    d_bar = np.mean(d)
    s_d = np.std(d, ddof=1)
    se_d = s_d / np.sqrt(n)
    df_val = n - 1

    t_stat = d_bar / se_d
    t_crit = get_t_critical(ALPHA, df_val, tail='two')
    p_value = calculate_p_value_t(t_stat, df_val, tail='two')
    reject_null = abs(t_stat) > t_crit

    t_scipy, p_scipy = stats.ttest_rel(rhs, lhs)

    results = {
        'n': n, 'df': df_val,
        'd_bar': d_bar, 's_d': s_d, 'se_d': se_d,
        'mean_rhs': np.mean(rhs), 'mean_lhs': np.mean(lhs),
        't_stat': t_stat, 't_crit': t_crit,
        'p_value': p_value, 'reject_null': reject_null,
        't_scipy': t_scipy, 'p_scipy': p_scipy,
    }

    return results


# ============================================================
# PART (c): PAIRED T-TEST — MALE RIGHT vs LEFT HAND SPAN
# ============================================================

def run_part_c(df):
    """
    Paired t-test comparing male right-hand span vs left-hand span.
    """
    mal = df[df["Biological Sex"] == "Male"].copy()
    rhs = mal["Right Hand Span (cm)"]
    lhs = mal["Left Hand Span (cm)"]
    mask = rhs.notna() & lhs.notna()
    rhs = rhs[mask].to_numpy()
    lhs = lhs[mask].to_numpy()

    d = rhs - lhs
    n = len(d)
    d_bar = np.mean(d)
    s_d = np.std(d, ddof=1)
    se_d = s_d / np.sqrt(n)
    df_val = n - 1

    t_stat = d_bar / se_d
    t_crit = get_t_critical(ALPHA, df_val, tail='two')
    p_value = calculate_p_value_t(t_stat, df_val, tail='two')
    reject_null = abs(t_stat) > t_crit

    t_scipy, p_scipy = stats.ttest_rel(rhs, lhs)

    results = {
        'n': n, 'df': df_val,
        'd_bar': d_bar, 's_d': s_d, 'se_d': se_d,
        'mean_rhs': np.mean(rhs), 'mean_lhs': np.mean(lhs),
        't_stat': t_stat, 't_crit': t_crit,
        'p_value': p_value, 'reject_null': reject_null,
        't_scipy': t_scipy, 'p_scipy': p_scipy,
    }

    return results


# ============================================================
# PART (d): TWO-SAMPLE POOLED T-TEST — ATHLETES/MUSICIANS vs NEITHER
# ============================================================

def classify_athlete_musician(row):
    """
    Classify a row as 'athlete/musician' or 'neither'.
    Non-sport entries: blank, 'N/A', 'None', 'Gym'.
    Non-music entries: blank, 'N/A', 'None', 'NA', 'none', 'n/a'.
    """
    non_sport = {"", "n/a", "none", "gym"}
    non_music = {"", "n/a", "none", "na"}

    has_sport = row["Sports"].lower() not in non_sport
    has_music = row["Music"].lower() not in non_music

    if has_sport or has_music:
        return "athlete_musician"
    else:
        return "neither"


def run_part_d(df):
    """
    Two-sample pooled t-test comparing mean height of
    athletes/musicians vs non-athletes/non-musicians.
    """
    df_d = df.copy()
    df_d["group"] = df_d.apply(classify_athlete_musician, axis=1)

    height_am = df_d.loc[
        (df_d["group"] == "athlete_musician") & df_d["Height (cm)"].notna(),
        "Height (cm)"
    ].to_numpy()

    height_neither = df_d.loc[
        (df_d["group"] == "neither") & df_d["Height (cm)"].notna(),
        "Height (cm)"
    ].to_numpy()

    n1, n2 = len(height_am), len(height_neither)
    x1_bar, x2_bar = np.mean(height_am), np.mean(height_neither)
    s1, s2 = np.std(height_am, ddof=1), np.std(height_neither, ddof=1)

    df_val = n1 + n2 - 2
    s_pooled_sq = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df_val
    t_stat = (x1_bar - x2_bar) / np.sqrt(s_pooled_sq * (1/n1 + 1/n2))

    t_crit = get_t_critical(ALPHA, df_val, tail='two')
    p_value = calculate_p_value_t(t_stat, df_val, tail='two')
    reject_null = abs(t_stat) > t_crit

    t_scipy, p_scipy = stats.ttest_ind(height_am, height_neither, equal_var=True)

    results = {
        'n1': n1, 'n2': n2,
        'x1_bar': x1_bar, 'x2_bar': x2_bar,
        's1': s1, 's2': s2,
        's_pooled_sq': s_pooled_sq,
        's_pooled': np.sqrt(s_pooled_sq),
        'df': df_val,
        't_stat': t_stat, 't_crit': t_crit,
        'p_value': p_value, 'reject_null': reject_null,
        't_scipy': t_scipy, 'p_scipy': p_scipy,
    }

    return results


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def display_two_sample(label, results, group1_name, group2_name):
    """
    Display results for a two-sample pooled t-test.
    """
    print(f"\n{'=' * 62}")
    print(f"PROBLEM 3{label}: COMPUTED RESULTS")
    print(f"{'=' * 62}")
    print(f"  {group1_name}:")
    print(f"    n1 = {results['n1']},  x̄1 = {results['x1_bar']:.4f},  s1 = {results['s1']:.4f}")
    print(f"  {group2_name}:")
    print(f"    n2 = {results['n2']},  x̄2 = {results['x2_bar']:.4f},  s2 = {results['s2']:.4f}")
    print(f"-" * 62)
    print(f"  s_pooled               =  {results['s_pooled']:.4f}")
    print(f"  df                     =  {results['df']}")
    print(f"  t_stat                 =  {results['t_stat']:.4f}")
    print(f"  t_crit (alpha=0.05, 2-tail)=  ±{results['t_crit']:.4f}")
    print(f"  p-value                =  {results['p_value']:.4e}")
    print(f"  Reject null?              {'Yes' if results['reject_null'] else 'No'}")
    print(f"-" * 62)
    print(f"  scipy t (verify)       =  {results['t_scipy']:.4f}")
    print(f"  scipy p (verify)       =  {results['p_scipy']:.4e}")
    print(f"{'=' * 62}\n")


def display_paired(label, results, var_name):
    """
    Display results for a paired t-test.
    """
    print(f"\n{'=' * 62}")
    print(f"PROBLEM 3{label}: COMPUTED RESULTS")
    print(f"{'=' * 62}")
    print(f"  {var_name}")
    print(f"    n = {results['n']},  df = {results['df']}")
    print(f"    Mean RHS = {results['mean_rhs']:.4f} cm")
    print(f"    Mean LHS = {results['mean_lhs']:.4f} cm")
    print(f"    d̄ (RHS - LHS) = {results['d_bar']:.4f} cm")
    print(f"    s_d = {results['s_d']:.4f},  SE_d = {results['se_d']:.4f}")
    print(f"-" * 62)
    print(f"  t_stat                 =  {results['t_stat']:.4f}")
    print(f"  t_crit (alpha=0.05, 2-tail)=  ±{results['t_crit']:.4f}")
    print(f"  p-value                =  {results['p_value']:.4e}")
    print(f"  Reject null?              {'Yes' if results['reject_null'] else 'No'}")
    print(f"-" * 62)
    print(f"  scipy t (verify)       =  {results['t_stat']:.4f}")
    print(f"  scipy p (verify)       =  {results['p_scipy']:.4e}")
    print(f"{'=' * 62}\n")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# BIOSTATISTICS HW3 — PROBLEM 3")
    print("#" * 70)

    df = load_and_clean(CSV_PATH)
    print(f"\n  Loaded {len(df)} total observations.")

    results_a = run_part_a(df)
    display_two_sample("(a)", results_a, "Male (avg hand span)", "Female (avg hand span)")

    results_b = run_part_b(df)
    display_paired("(b)", results_b, "Females: Right vs Left Hand Span")

    results_c = run_part_c(df)
    display_paired("(c)", results_c, "Males: Right vs Left Hand Span")

    results_d = run_part_d(df)
    display_two_sample("(d)", results_d, "Athletes/Musicians (height)", "Neither (height)")

    print("\nCOMPUTATION COMPLETE")
    print("=" * 70 + "\n")
