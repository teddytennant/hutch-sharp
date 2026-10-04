"""Console demo for one-sided Hutchinson, Hutch#, and sketch-reuse weighted Hutch#."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from hutch_sharp import (
    hutch_sharp_matrix,
    one_sided_hutchinson_matrix,
    weighted_hutch_sharp_sketch_alpha_matrix,
)


def relative_rmse(estimates, truth):
    estimates = jnp.asarray(estimates, dtype=jnp.float64)
    err = estimates - truth
    return jnp.sqrt(jnp.mean(err * err)) / truth


def _trial_estimates(key, A, r_hutch, r_sharp):
    k_h, k_s, k_w = jax.random.split(key, 3)
    hutch = one_sided_hutchinson_matrix(A, r_hutch, key=k_h)
    sharp = hutch_sharp_matrix(A, r_sharp, key=k_s)
    weighted, _alpha = weighted_hutch_sharp_sketch_alpha_matrix(A, r_sharp, key=k_w)
    return hutch, sharp, weighted


def run_case(singular_values, budget, n_trials, key):
    """Relative RMSE at a fixed matvec budget.

    Hutchinson uses r = budget products with A only.
    Hutch# and sketch-reuse weighted Hutch# use r = budget/2, so products
    with A and A.T together equal the budget.
    """
    A = jnp.diag(jnp.asarray(singular_values, dtype=jnp.float64))
    truth = jnp.sum(jnp.asarray(singular_values, dtype=jnp.float64) ** 2)
    r_sharp = budget // 2
    keys = jax.random.split(key, n_trials)
    hutch, sharp, weighted = jax.vmap(
        lambda k: _trial_estimates(k, A, budget, r_sharp)
    )(keys)
    return (
        float(relative_rmse(hutch, truth)),
        float(relative_rmse(sharp, truth)),
        float(relative_rmse(weighted, truth)),
    )


def main():
    n = 64
    n_trials = 40
    budgets = (16, 32, 64)
    exponents = (0.5, 2.0)
    indices = jnp.arange(1, n + 1, dtype=jnp.float64)
    key = jax.random.PRNGKey(0)

    print("relative RMSE, n=64, 40 trials, seed 0")
    print(f"{'c':>6} {'budget':>8} {'Hutchinson':>12} {'Hutch#':>12} {'weighted':>12}")
    for exponent in exponents:
        singular_values = indices ** (-exponent)
        for budget in budgets:
            key, sub = jax.random.split(key)
            hutch, sharp, weighted = run_case(singular_values, budget, n_trials, sub)
            print(
                f"{exponent:6.1f} {budget:8d} {hutch:12.6e} {sharp:12.6e} {weighted:12.6e}"
            )


if __name__ == "__main__":
    main()
