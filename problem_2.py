"""
Biostatistics Homework 3 - Problem 2
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
    Load the hand span dataset and return cleaned height and
    right-hand-span arrays with NaN/invalid rows dropped.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    height = pd.to_numeric(df["Height (cm)"], errors="coerce")
    rhs = pd.to_numeric(df["Right Hand Span (cm)"], errors="coerce")

    mask = height.notna() & rhs.notna()
    height = height[mask].to_numpy()
    rhs = rhs[mask].to_numpy()

    return height, rhs


# ============================================================
# PROBLEM 2(a): PEARSON CORRELATION + T-TEST
# ============================================================

def calculate_pearson_r(x, y):
    """
    Calculate the Pearson correlation coefficient.

    r = sum((xi - x_bar)(yi - y_bar)) /
        sqrt(sum((xi - x_bar)^2) * sum((yi - y_bar)^2))
    """
    x_bar = np.mean(x)
    y_bar = np.mean(y)

    numerator = np.sum((x - x_bar) * (y - y_bar))
    denominator = np.sqrt(np.sum((x - x_bar)**2) * np.sum((y - y_bar)**2))

    r = numerator / denominator
    return r


def calculate_t_from_r(r, n):
    """
    Convert Pearson r to a t test statistic.

    t = r * sqrt(n - 2) / sqrt(1 - r^2)
    """
    t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
    return t


def get_t_critical(alpha, df, tail='two'):
    """
    Get the critical t-value for the given alpha, df, and tail direction.
    """
    if tail == 'two':
        t_crit = stats.t.ppf(1 - alpha / 2, df)
    elif tail == 'right':
        t_crit = stats.t.ppf(1 - alpha, df)
    elif tail == 'left':
        t_crit = stats.t.ppf(alpha, df)
    else:
        raise ValueError(f"Invalid tail type: {tail}")

    return t_crit


def calculate_p_value_t(t, df, tail='two'):
    """
    Calculate the p-value for the given t-score and tail direction.
    """
    if tail == 'two':
        p = 2 * (1 - stats.t.cdf(abs(t), df))
    elif tail == 'right':
        p = 1 - stats.t.cdf(t, df)
    elif tail == 'left':
        p = stats.t.cdf(t, df)
    else:
        raise ValueError(f"Invalid tail type: {tail}")

    return p


def run_correlation_test(height, rhs):
    """
    Execute the 9-step hypothesis testing procedure for Problem 2(a).
    Tests whether height and right-hand span are correlated.
    """
    n = len(height)
    df = n - 2

    r = calculate_pearson_r(height, rhs)
    t_stat = calculate_t_from_r(r, n)
    t_crit = get_t_critical(ALPHA, df, tail='two')
    p_value = calculate_p_value_t(t_stat, df, tail='two')
    reject_null = abs(t_stat) > t_crit
    r_scipy, p_scipy = stats.pearsonr(height, rhs)

    results = {
        'n': n,
        'df': df,
        'alpha': ALPHA,
        'r': r,
        't_stat': t_stat,
        't_crit': t_crit,
        'p_value': p_value,
        'reject_null': reject_null,
        'r_scipy': r_scipy,
        'p_scipy': p_scipy,
        'x_bar_height': np.mean(height),
        'x_bar_rhs': np.mean(rhs),
        'sd_height': np.std(height, ddof=1),
        'sd_rhs': np.std(rhs, ddof=1),
    }

    return results


# ============================================================
# FIGURE
# ============================================================

def save_scatter_plot(height, rhs, results, fig_dir):
    """
    Generate and save a scatter plot of height vs. right-hand span
    with a linear regression line.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(height, rhs, alpha=0.5, edgecolors='k', linewidths=0.5, s=40)

    slope, intercept = np.polyfit(height, rhs, 1)
    x_line = np.linspace(height.min(), height.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color='r', linewidth=1.5,
            label=f"y = {slope:.4f}x + {intercept:.2f}")

    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Right Hand Span (cm)")
    ax.set_title("Height vs. Right Hand Span")

    r = results['r']
    p = results['p_value']
    ax.legend(title=f"r = {r:.4f}, p = {p:.4e}", loc='upper left')

    fig.tight_layout()
    path = f"{fig_dir}/problem_2_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved: {path}")


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def display_results(results):
    """
    Display computed results.
    """
    print("\n" + "=" * 58)
    print("PROBLEM 2(a): COMPUTED RESULTS")
    print("=" * 58)
    print(f"  n                          =  {results['n']}")
    print(f"  df                         =  {results['df']}")
    print(f"  Mean height                =  {results['x_bar_height']:.2f} cm")
    print(f"  Mean right hand span       =  {results['x_bar_rhs']:.4f} cm")
    print(f"  SD height                  =  {results['sd_height']:.4f} cm")
    print(f"  SD right hand span         =  {results['sd_rhs']:.4f} cm")
    print("-" * 58)
    print(f"  Pearson r                  =  {results['r']:.4f}")
    print(f"  t_stat                     =  {results['t_stat']:.4f}")
    print(f"  t_crit (alpha=0.05, 2-tail)=  ±{results['t_crit']:.4f}")
    print(f"  p-value                    =  {results['p_value']:.4e}")
    print(f"  Reject null?                  {'Yes' if results['reject_null'] else 'No'}")
    print("-" * 58)
    print(f"  scipy r (verify)           =  {results['r_scipy']:.4f}")
    print(f"  scipy p (verify)           =  {results['p_scipy']:.4e}")
    print("=" * 58 + "\n")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# BIOSTATISTICS HW3 — PROBLEM 2")
    print("#" * 70)

    height, rhs = load_and_clean(CSV_PATH)
    print(f"\n  Loaded {len(height)} valid observations.")

    results = run_correlation_test(height, rhs)
    display_results(results)

    save_scatter_plot(height, rhs, results, FIG_DIR)

    print("\nCOMPUTATION COMPLETE")
    print("=" * 70 + "\n")
