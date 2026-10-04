import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from hutch_sharp import (
    alpha_hat,
    alpha_hat_from_A_omega,
    hutch_flat,
    hutch_flat_explicit,
    hutch_flat_matrix,
    hutch_sharp,
    hutch_sharp_matrix,
    matvec_rmatvec_from_matrix,
    one_sided_hutchinson,
    optimal_alpha,
    optimal_alpha_from_singular_values,
    symmetrized_hutchinson,
    symmetrized_hutchinson_matrix,
    weighted_hutch_sharp,
    weighted_hutch_sharp_matrix,
    weighted_hutch_sharp_sketch_alpha,
)


def _rel(a, b):
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    return jnp.abs(a - b) / jnp.maximum(jnp.abs(b), 1e-30)


def assert_rel(a, b, tol=1e-9):
    err = _rel(a, b)
    worst = float(jnp.max(err))
    assert worst <= tol, f"relative error {worst} exceeds {tol}"


class _Counter:
    def __init__(self, A):
        self.A = jnp.asarray(A, dtype=jnp.float64)
        self.matvec_batches = []
        self.rmatvec_batches = []

    def matvec(self, V):
        V = jnp.asarray(V, dtype=jnp.float64)
        self.matvec_batches.append(int(V.shape[1]))
        return self.A @ V

    def rmatvec(self, U):
        U = jnp.asarray(U, dtype=jnp.float64)
        self.rmatvec_batches.append(int(U.shape[1]))
        return self.A.T @ U


def test_hutch_sharp_matches_equation_2_rectangular():
    key = jax.random.PRNGKey(1)
    kA, kO, kP = jax.random.split(key, 3)
    m, n, r = 5, 8, 4
    A = jax.random.normal(kA, (m, n), dtype=jnp.float64)
    omega = jax.random.normal(kO, (n, r), dtype=jnp.float64)
    psi = jax.random.normal(kP, (m, r), dtype=jnp.float64)

    matvec, rmatvec = matvec_rmatvec_from_matrix(A)
    est = hutch_sharp(matvec, rmatvec, m, n, r, omega=omega, psi=psi)
    est_matrix = hutch_sharp_matrix(A, r, omega=omega, psi=psi)

    A_omega = A @ omega
    AT_psi = A.T @ psi
    cross = A_omega.T @ psi
    expected = (
        jnp.sum(A_omega * A_omega) / r
        + jnp.sum(AT_psi * AT_psi) / r
        - jnp.sum(cross * cross) / (r * r)
    )
    assert_rel(est, expected)
    assert_rel(est_matrix, expected)
    assert_rel(jnp.sum(cross * cross), jnp.sum((omega.T @ AT_psi) ** 2))


def test_weighted_endpoints_match_on_same_sketches():
    key = jax.random.PRNGKey(2)
    kA, kO, kP = jax.random.split(key, 3)
    m, n, r = 6, 4, 5
    A = jax.random.normal(kA, (m, n), dtype=jnp.float64)
    omega = jax.random.normal(kO, (n, r), dtype=jnp.float64)
    psi = jax.random.normal(kP, (m, r), dtype=jnp.float64)

    sharp = hutch_sharp_matrix(A, r, omega=omega, psi=psi)
    sym = symmetrized_hutchinson_matrix(A, r, omega=omega, psi=psi)
    w1 = weighted_hutch_sharp_matrix(A, r, 1.0, omega=omega, psi=psi)
    w0 = weighted_hutch_sharp_matrix(A, r, 0.0, omega=omega, psi=psi)
    assert_rel(w1, sharp)
    assert_rel(w0, sym)


def test_weighted_equals_convex_combination():
    key = jax.random.PRNGKey(3)
    kA, kO, kP = jax.random.split(key, 3)
    m, n, r = 7, 5, 3
    alpha = 0.37
    A = jax.random.normal(kA, (m, n), dtype=jnp.float64)
    omega = jax.random.normal(kO, (n, r), dtype=jnp.float64)
    psi = jax.random.normal(kP, (m, r), dtype=jnp.float64)

    sharp = hutch_sharp_matrix(A, r, omega=omega, psi=psi)
    sym = symmetrized_hutchinson_matrix(A, r, omega=omega, psi=psi)
    weighted = weighted_hutch_sharp_matrix(A, r, alpha, omega=omega, psi=psi)
    combo = (1.0 - alpha) * sym + alpha * sharp
    assert_rel(weighted, combo)

    matvec, rmatvec = matvec_rmatvec_from_matrix(A)
    weighted_op = weighted_hutch_sharp(
        matvec, rmatvec, m, n, r, alpha, omega=omega, psi=psi
    )
    assert_rel(weighted_op, combo)


def test_optimal_alpha_on_diagonal():
    diagonal = jnp.asarray([1.5, -2.0, 0.25, 3.0, -0.5], dtype=jnp.float64)
    singular_values = jnp.abs(diagonal)
    r = 6
    fro_sq = jnp.sum(singular_values ** 2)
    sch4 = jnp.sum(singular_values ** 4)
    R = (fro_sq ** 2) / sch4
    T = (R + 1.0) / r
    expected = 1.0 / (1.0 + 2.0 * T)
    assert_rel(optimal_alpha(fro_sq, sch4, r), expected)
    assert_rel(optimal_alpha_from_singular_values(singular_values, r), expected)

    # Singular values of a diagonal matrix are the absolute diagonal entries.
    A = jnp.diag(diagonal)
    svd_s = jnp.linalg.svd(A, compute_uv=False)
    assert_rel(jnp.sort(svd_s), jnp.sort(singular_values))


def test_alpha_hat_reuses_provided_omega_split_in_half():
    key = jax.random.PRNGKey(4)
    kA, kO = jax.random.split(key)
    m, n, r = 5, 6, 8
    A = jax.random.normal(kA, (m, n), dtype=jnp.float64)
    omega = jax.random.normal(kO, (n, r), dtype=jnp.float64)
    half = r // 2
    Y = A @ omega
    fro_sq = jnp.sum(Y * Y)
    F_hat = (fro_sq ** 2) / (r ** 2)
    gram = Y[:, :half].T @ Y[:, half:]
    S_hat = (4.0 / (r ** 2)) * jnp.sum(gram * gram)
    T_hat = (F_hat / S_hat + 1.0) / r
    expected = 1.0 / (1.0 + 2.0 * T_hat)

    calls = []

    def matvec(V):
        calls.append(jnp.asarray(V))
        return A @ V

    got, A_omega, used = alpha_hat(matvec, n, r, omega=omega)
    assert_rel(got, expected)
    assert_rel(alpha_hat_from_A_omega(Y), expected)
    assert len(calls) == 1
    assert calls[0].shape == (n, r)
    assert_rel(jnp.max(jnp.abs(used - omega)), 0.0, tol=0.0)
    assert_rel(jnp.max(jnp.abs(A_omega - Y)), 0.0, tol=0.0)

    other = jax.random.normal(jax.random.PRNGKey(99), (n, r), dtype=jnp.float64)
    other_alpha, _, _ = alpha_hat(matvec, n, r, omega=other)
    assert float(jnp.abs(other_alpha - got)) > 1e-8

    psi = jax.random.normal(jax.random.PRNGKey(7), (m, r), dtype=jnp.float64)
    _est, alpha_used = weighted_hutch_sharp_sketch_alpha(
        matvec, lambda U: A.T @ U, m, n, r, omega=omega, psi=psi
    )
    assert_rel(alpha_used, expected)


def test_alpha_hat_zero_sketch_edges():
    # Documented edge, not specified by the paper.
    r = 4
    zero = jnp.zeros((3, r), dtype=jnp.float64)
    assert float(alpha_hat_from_A_omega(zero)) == 1.0

    # S_hat == 0 and F_hat > 0: the two halves are orthogonal in the row space.
    Y = jnp.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    assert float(alpha_hat_from_A_omega(Y)) == 0.0


def test_hutch_flat_matches_explicit_pinv():
    key = jax.random.PRNGKey(5)
    kA, kO, kP, kG = jax.random.split(key, 4)
    m, n, r = 6, 5, 2
    A = jax.random.normal(kA, (m, n), dtype=jnp.float64)
    omega = jax.random.normal(kO, (n, 2 * r), dtype=jnp.float64)
    psi = jax.random.normal(kP, (m, 4 * r), dtype=jnp.float64)
    G = jax.random.normal(kG, (n, 2 * r), dtype=jnp.float64)

    est = hutch_flat_matrix(A, r, omega=omega, psi=psi, G=G)
    ref = hutch_flat_explicit(A, omega, psi, G)

    Y = A @ omega
    C = psi.T @ Y
    M = jax.numpy.linalg.pinv(C)
    B = Y @ M @ (psi.T @ A)
    AG = A @ G
    BG = B @ G
    manual = jnp.sum(B * B) + (jnp.sum(AG * AG) - jnp.sum(BG * BG)) / (2.0 * r)
    assert_rel(est, ref)
    assert_rel(est, manual)
    assert_rel(ref, manual)

    matvec, rmatvec = matvec_rmatvec_from_matrix(A)
    est_op = hutch_flat(matvec, rmatvec, m, n, r, omega=omega, psi=psi, G=G)
    assert_rel(est_op, manual)


def test_zero_matrix_estimators_are_zero():
    m, n, r = 4, 3, 2
    A = jnp.zeros((m, n), dtype=jnp.float64)
    key = jax.random.PRNGKey(6)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    omega = jax.random.normal(k1, (n, r), dtype=jnp.float64)
    psi = jax.random.normal(k2, (m, r), dtype=jnp.float64)
    values = [
        one_sided_hutchinson(lambda V: A @ V, n, r, omega=omega),
        symmetrized_hutchinson(
            lambda V: A @ V, lambda U: A.T @ U, m, n, r, omega=omega, psi=psi
        ),
        hutch_sharp(
            lambda V: A @ V, lambda U: A.T @ U, m, n, r, omega=omega, psi=psi
        ),
        weighted_hutch_sharp(
            lambda V: A @ V,
            lambda U: A.T @ U,
            m,
            n,
            r,
            0.4,
            omega=omega,
            psi=psi,
        ),
        weighted_hutch_sharp_sketch_alpha(
            lambda V: A @ V, lambda U: A.T @ U, m, n, r, omega=omega, psi=psi
        )[0],
        hutch_flat(
            lambda V: A @ V,
            lambda U: A.T @ U,
            m,
            n,
            r,
            key=k3,
        ),
        hutch_flat_matrix(A, r, key=k4),
    ]
    for value in values:
        assert float(jnp.abs(value)) == 0.0


def test_nonadaptive_column_counts():
    m, n, r = 5, 7, 4
    A = jax.random.normal(jax.random.PRNGKey(8), (m, n), dtype=jnp.float64)
    key = jax.random.PRNGKey(9)

    counter = _Counter(A)
    hutch_sharp(counter.matvec, counter.rmatvec, m, n, r, key=key)
    assert counter.matvec_batches == [r]
    assert counter.rmatvec_batches == [r]

    counter = _Counter(A)
    hutch_flat(counter.matvec, counter.rmatvec, m, n, r, key=key)
    assert sum(counter.matvec_batches) == 4 * r
    assert sum(counter.rmatvec_batches) == 4 * r
    assert counter.matvec_batches == [4 * r]
    assert counter.rmatvec_batches == [4 * r]


def _mc_draws(n_trials, n, r, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), n_trials)
    omega = jax.vmap(lambda k: jax.random.normal(k, (n, r), dtype=jnp.float64))(
        jax.vmap(lambda k: jax.random.split(k, 2)[0])(keys)
    )
    psi = jax.vmap(lambda k: jax.random.normal(k, (n, r), dtype=jnp.float64))(
        jax.vmap(lambda k: jax.random.split(k, 2)[1])(keys)
    )
    return omega, psi


def test_monte_carlo_hutch_sharp_mean_and_variance():
    diagonal = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    A = jnp.diag(diagonal)
    n = 4
    r = 8
    n_trials = 4000
    truth = float(jnp.sum(diagonal ** 2))
    fro4 = truth ** 2
    sch4 = float(jnp.sum(diagonal ** 4))
    theory = (2.0 / r**2) * sch4 + (2.0 / r**2) * fro4

    omega, psi = _mc_draws(n_trials, n, r, seed=0)
    estimates = jax.vmap(lambda o, p: hutch_sharp_matrix(A, r, omega=o, psi=p))(
        omega, psi
    )
    mean = float(jnp.mean(estimates))
    sample_var = float(jnp.var(estimates, ddof=1))
    mean_ratio = mean / truth
    var_ratio = sample_var / theory
    assert abs(mean_ratio - 1.0) < 0.03, f"mean ratio {mean_ratio}"
    assert abs(var_ratio - 1.0) < 0.30, f"sample/theory variance ratio {var_ratio}"


def test_monte_carlo_weighted_optimal_alpha():
    diagonal = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    A = jnp.diag(diagonal)
    n = 4
    r = 8
    n_trials = 4000
    truth = float(jnp.sum(diagonal ** 2))
    fro4 = truth ** 2
    sch4 = float(jnp.sum(diagonal ** 4))
    R = fro4 / sch4
    T = (R + 1.0) / r
    alpha = 1.0 / (1.0 + 2.0 * T)
    theory = ((1.0 - alpha) ** 2) / r * sch4 + (2.0 * alpha**2) / (r**2) * (
        sch4 + fro4
    )

    omega, psi = _mc_draws(n_trials, n, r, seed=0)
    estimates = jax.vmap(
        lambda o, p: weighted_hutch_sharp_matrix(A, r, alpha, omega=o, psi=p)
    )(omega, psi)
    mean = float(jnp.mean(estimates))
    sample_var = float(jnp.var(estimates, ddof=1))
    mean_ratio = mean / truth
    var_ratio = sample_var / theory
    assert abs(mean_ratio - 1.0) < 0.03, f"mean ratio {mean_ratio}"
    assert abs(var_ratio - 1.0) < 0.35, f"sample/theory variance ratio {var_ratio}"


def test_alpha_hat_rejects_odd_r():
    Y = jnp.ones((2, 3), dtype=jnp.float64)
    with pytest.raises(ValueError):
        alpha_hat_from_A_omega(Y)
