"""
Biostatistics Homework 3 - Problem 1
Author: Logan Kaising
Date: March 2026
Software: Python 3.11.13 with numpy, scipy
"""

import numpy as np
from scipy import stats


# ============================================================
# GIVEN PARAMETERS
# ============================================================

MU_0 = 61500        # Population mean (USDoE reported), $
SIGMA = 15850       # Population standard deviation (known), $
N = 23              # Sample size
X_BAR = 68300       # Sample mean (Rutgers PhDs), $
ALPHA = 0.05        # Significance level


# ============================================================
# PROBLEM 1
# ============================================================

def calculate_standard_error(sigma, n):
    """
    Calculate the standard error of the mean.

    SE = sigma / sqrt(n)
    """
    se = sigma / np.sqrt(n)
    return se


def calculate_z_score(x_bar, mu_0, se):
    """
    Calculate the z test statistic for a one-sample z-test.

    z = (x_bar - mu_0) / SE
    """
    z = (x_bar - mu_0) / se
    return z


def get_z_critical(alpha, tail='right'):
    """
    Get the critical z-value for the given alpha and tail direction.
    """
    if tail == 'right':
        z_crit = stats.norm.ppf(1 - alpha)
    elif tail == 'left':
        z_crit = stats.norm.ppf(alpha)
    elif tail == 'two':
        z_crit = stats.norm.ppf(1 - alpha / 2)
    else:
        raise ValueError(f"Invalid tail type: {tail}")

    return z_crit


def calculate_p_value(z, tail='right'):
    """
    Calculate the p-value for the given z-score and tail direction.
    """
    if tail == 'right':
        p = 1 - stats.norm.cdf(z)
    elif tail == 'left':
        p = stats.norm.cdf(z)
    elif tail == 'two':
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        raise ValueError(f"Invalid tail type: {tail}")

    return p


def run_hypothesis_test():
    """
    Execute the 9-step hypothesis testing procedure for Problem 1.
    """
    z_crit = get_z_critical(ALPHA, tail='right')
    se = calculate_standard_error(SIGMA, N)
    z = calculate_z_score(X_BAR, MU_0, se)
    p_value = calculate_p_value(z, tail='right')
    reject_null = z > z_crit

    results = {
        'mu_0': MU_0,
        'sigma': SIGMA,
        'n': N,
        'x_bar': X_BAR,
        'alpha': ALPHA,
        'se': se,
        'z_stat': z,
        'z_crit': z_crit,
        'p_value': p_value,
        'reject_null': reject_null,
    }

    return results


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def display_results(results):
    """
    Display computed results.
    """
    print("\n" + "=" * 50)
    print("PROBLEM 1: COMPUTED RESULTS")
    print("=" * 50)
    print(f"  SE  = sigma / sqrt(n)  = ${results['se']:,.2f}")
    print(f"  z_stat                 =  {results['z_stat']:.4f}")
    print(f"  z_crit (alpha=0.05)    =  {results['z_crit']:.4f}")
    print(f"  p-value                =  {results['p_value']:.6f}")
    print(f"  Reject null?              {'Yes' if results['reject_null'] else 'No'}")
    print("=" * 50 + "\n")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# BIOSTATISTICS HW3 — PROBLEM 1")
    print("#" * 70)

    results = run_hypothesis_test()
    display_results(results)

    print("COMPUTATION COMPLETE")
    print("=" * 70 + "\n")
