"""Non-adaptive Frobenius norm estimators from Hutch# (arXiv 2609.28472)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _as_float64(x):
    return jnp.asarray(x, dtype=jnp.float64)


def frobenius_sq(X):
    """Squared Frobenius norm, sum of squares."""
    X = _as_float64(X)
    return jnp.sum(X * X)


def gaussian_sketch(key, rows, cols):
    """i.i.d. standard Gaussian sketch of shape (rows, cols)."""
    return jax.random.normal(key, (rows, cols), dtype=jnp.float64)


def _split_key(key, n):
    if key is None:
        raise ValueError("a PRNG key is required when a sketch is not provided")
    return jax.random.split(key, n)


def _resolve_one_sketch(key, shape, sketch):
    if sketch is not None:
        return _as_float64(sketch)
    keys = _split_key(key, 1)
    return gaussian_sketch(keys[0], shape[0], shape[1])


def _resolve_two_sketches(key, shape_omega, shape_psi, omega, psi):
    if omega is not None and psi is not None:
        return _as_float64(omega), _as_float64(psi)
    if omega is None and psi is None:
        k1, k2 = _split_key(key, 2)
        return gaussian_sketch(k1, *shape_omega), gaussian_sketch(k2, *shape_psi)
    if omega is None:
        k1 = _split_key(key, 1)[0]
        return gaussian_sketch(k1, *shape_omega), _as_float64(psi)
    k1 = _split_key(key, 1)[0]
    return _as_float64(omega), gaussian_sketch(k1, *shape_psi)


def matvec_rmatvec_from_matrix(A):
    """Thin wrappers: matvec(V) = A @ V, rmatvec(U) = A.T @ U."""
    A = _as_float64(A)

    def matvec(V):
        return A @ _as_float64(V)

    def rmatvec(U):
        return A.T @ _as_float64(U)

    return matvec, rmatvec


def one_sided_hutchinson(matvec, n, r, key=None, omega=None):
    """Girard-Hutchinson: (1/r) ||A Omega||_F^2. matvec only.

    Omega has shape (n, r). Pass a pre-drawn sketch or a PRNG key.
    """
    omega = _resolve_one_sketch(key, (n, r), omega)
    A_omega = matvec(omega)
    return frobenius_sq(A_omega) / r


def one_sided_hutchinson_matrix(A, r, key=None, omega=None):
    """One-sided Hutchinson for an explicit array A."""
    A = _as_float64(A)
    matvec, _ = matvec_rmatvec_from_matrix(A)
    return one_sided_hutchinson(matvec, A.shape[1], r, key=key, omega=omega)


def symmetrized_hutchinson(matvec, rmatvec, m, n, r, key=None, omega=None, psi=None):
    """Symmetrized Hutchinson H_r(A) = (1/(2r)) (||A Omega||_F^2 + ||Psi^T A||_F^2).

    Omega is (n, r), Psi is (m, r). ||Psi^T A||_F^2 is computed as ||A.T Psi||_F^2.
    """
    omega, psi = _resolve_two_sketches(key, (n, r), (m, r), omega, psi)
    A_omega = matvec(omega)
    AT_psi = rmatvec(psi)
    return (frobenius_sq(A_omega) + frobenius_sq(AT_psi)) / (2.0 * r)


def symmetrized_hutchinson_matrix(A, r, key=None, omega=None, psi=None):
    """Symmetrized Hutchinson for an explicit array A."""
    A = _as_float64(A)
    matvec, rmatvec = matvec_rmatvec_from_matrix(A)
    m, n = A.shape
    return symmetrized_hutchinson(
        matvec, rmatvec, m, n, r, key=key, omega=omega, psi=psi
    )


def _sketch_products(matvec, rmatvec, omega, psi):
    """One batched matvec and one batched rmatvec, plus the cross term.

    Psi^T A Omega = (A Omega).T @ Psi, so the cross term needs no extra product.
    """
    A_omega = matvec(omega)
    AT_psi = rmatvec(psi)
    cross = A_omega.T @ psi
    return A_omega, AT_psi, cross


def hutch_sharp(matvec, rmatvec, m, n, r, key=None, omega=None, psi=None):
    """Hutch#, equation (2):

    H_r#(A) = (1/r) ||A Omega||_F^2 + (1/r) ||Psi^T A||_F^2
              - (1/r^2) ||Psi^T A Omega||_F^2.

    Non-adaptive: one batched matvec of r columns and one batched rmatvec of r columns.
    """
    omega, psi = _resolve_two_sketches(key, (n, r), (m, r), omega, psi)
    A_omega, AT_psi, cross = _sketch_products(matvec, rmatvec, omega, psi)
    r = jnp.asarray(r, dtype=jnp.float64)
    return (
        frobenius_sq(A_omega) / r
        + frobenius_sq(AT_psi) / r
        - frobenius_sq(cross) / (r * r)
    )


def hutch_sharp_matrix(A, r, key=None, omega=None, psi=None):
    """Hutch# for an explicit array A."""
    A = _as_float64(A)
    matvec, rmatvec = matvec_rmatvec_from_matrix(A)
    m, n = A.shape
    return hutch_sharp(matvec, rmatvec, m, n, r, key=key, omega=omega, psi=psi)


def weighted_hutch_sharp(
    matvec, rmatvec, m, n, r, alpha, key=None, omega=None, psi=None
):
    """Weighted family, equation (4):

    H_{r,alpha}#(A) = ((1+alpha)/(2r)) ||A Omega||_F^2
                      + ((1+alpha)/(2r)) ||Psi^T A||_F^2
                      - (alpha/r^2) ||Psi^T A Omega||_F^2.

    On the same sketches this equals (1-alpha) H_r + alpha H_r#.
    alpha = 1 is Hutch#. alpha = 0 is symmetrized Hutchinson.
    """
    omega, psi = _resolve_two_sketches(key, (n, r), (m, r), omega, psi)
    A_omega, AT_psi, cross = _sketch_products(matvec, rmatvec, omega, psi)
    return weighted_from_products(A_omega, AT_psi, cross, r, alpha)


def weighted_from_products(A_omega, AT_psi, cross, r, alpha):
    """Equation (4) from already computed sketch products."""
    alpha = _as_float64(alpha)
    r = jnp.asarray(r, dtype=jnp.float64)
    scale = (1.0 + alpha) / (2.0 * r)
    return (
        scale * frobenius_sq(A_omega)
        + scale * frobenius_sq(AT_psi)
        - (alpha / (r * r)) * frobenius_sq(cross)
    )


def weighted_hutch_sharp_matrix(A, r, alpha, key=None, omega=None, psi=None):
    """Weighted Hutch# for an explicit array A."""
    A = _as_float64(A)
    matvec, rmatvec = matvec_rmatvec_from_matrix(A)
    m, n = A.shape
    return weighted_hutch_sharp(
        matvec, rmatvec, m, n, r, alpha, key=key, omega=omega, psi=psi
    )


def schatten_4_fourth(singular_values):
    """||A||_(4)^4 = sum_j sigma_j(A)^4."""
    s = _as_float64(singular_values)
    return jnp.sum(s ** 4)


def frobenius_fourth_from_sq(frobenius_sq_value):
    """||A||_F^4 = (||A||_F^2)^2."""
    v = _as_float64(frobenius_sq_value)
    return v * v


def optimal_alpha(frobenius_sq_value, schatten_4_fourth_value, r):
    """Optimal alpha from true norms.

    R = ||A||_F^4 / ||A||_(4)^4, T = (R + 1) / r, alpha* = 1 / (1 + 2 T).
    ||A||_F^4 means (||A||_F^2)^2.
    """
    fro4 = frobenius_fourth_from_sq(frobenius_sq_value)
    sch4 = _as_float64(schatten_4_fourth_value)
    r = jnp.asarray(r, dtype=jnp.float64)
    R = fro4 / sch4
    T = (R + 1.0) / r
    return 1.0 / (1.0 + 2.0 * T)


def optimal_alpha_from_singular_values(singular_values, r):
    """alpha* when singular values are known (e.g. a diagonal matrix)."""
    s = _as_float64(singular_values)
    fro_sq = jnp.sum(s * s)
    sch4 = schatten_4_fourth(s)
    return optimal_alpha(fro_sq, sch4, r)


def alpha_hat_from_A_omega(A_omega):
    """Practical alpha-hat from an existing A Omega sketch. No fresh draw.

    r must be even and at least 2.

    F_hat = (1/r^2) ||A Omega||_F^4 approximates ||A||_F^4 (equation 9).
    Split Omega into the first r/2 columns and the rest, which splits A Omega
    the same way. S_hat = (4/r^2) ||(A Omega_1).T @ (A Omega_2)||_F^2
    approximates ||A||_(4)^4 (equation 10).
    T_hat = (F_hat / S_hat + 1) / r, alpha_hat = 1 / (1 + 2 T_hat).

    Edge case, not specified by the paper: if S_hat == 0 and F_hat == 0,
    alpha_hat = 1. If S_hat == 0 and F_hat > 0, alpha_hat = 0.
    """
    A_omega = _as_float64(A_omega)
    r_int = int(A_omega.shape[1])
    if r_int < 2 or r_int % 2 != 0:
        raise ValueError("alpha-hat requires r even and at least 2")
    half = r_int // 2
    r = jnp.asarray(r_int, dtype=jnp.float64)
    fro_sq = frobenius_sq(A_omega)
    # ||A Omega||_F^4 = (||A Omega||_F^2)^2
    F_hat = (fro_sq * fro_sq) / (r * r)
    cross = A_omega[:, :half].T @ A_omega[:, half:]
    S_hat = (4.0 / (r * r)) * frobenius_sq(cross)
    safe_S = jnp.where(S_hat == 0.0, 1.0, S_hat)
    T_hat = (F_hat / safe_S + 1.0) / r
    alpha = 1.0 / (1.0 + 2.0 * T_hat)
    both_zero = (S_hat == 0.0) & (F_hat == 0.0)
    s_zero_f_pos = (S_hat == 0.0) & (F_hat > 0.0)
    alpha = jnp.where(both_zero, 1.0, alpha)
    alpha = jnp.where(s_zero_f_pos, 0.0, alpha)
    return alpha


def alpha_hat(matvec, n, r, key=None, omega=None):
    """alpha-hat reusing one A Omega sketch. Does not draw a second sketch."""
    omega = _resolve_one_sketch(key, (n, r), omega)
    A_omega = matvec(omega)
    return alpha_hat_from_A_omega(A_omega), A_omega, omega


def weighted_hutch_sharp_sketch_alpha(
    matvec, rmatvec, m, n, r, key=None, omega=None, psi=None
):
    """Weighted Hutch# with alpha-hat from the same A Omega used in the estimator.

    r must be even and at least 2. Psi is an independent sketch of shape (m, r)
    unless one is supplied. Omega is not redrawn for the alpha estimate.
    """
    omega, psi = _resolve_two_sketches(key, (n, r), (m, r), omega, psi)
    A_omega, AT_psi, cross = _sketch_products(matvec, rmatvec, omega, psi)
    alpha = alpha_hat_from_A_omega(A_omega)
    estimate = weighted_from_products(A_omega, AT_psi, cross, r, alpha)
    return estimate, alpha


def weighted_hutch_sharp_sketch_alpha_matrix(A, r, key=None, omega=None, psi=None):
    """Sketch-reuse weighted Hutch# for an explicit array A."""
    A = _as_float64(A)
    matvec, rmatvec = matvec_rmatvec_from_matrix(A)
    m, n = A.shape
    return weighted_hutch_sharp_sketch_alpha(
        matvec, rmatvec, m, n, r, key=key, omega=omega, psi=psi
    )


def hutch_flat(matvec, rmatvec, m, n, r, key=None, omega=None, psi=None, G=None):
    """Hutch-flat, Section 4. Sketch sizes are exactly (2r, 4r, 2r). Not retuned.

    Omega is (n, 2r), Psi is (m, 4r), G is (n, 2r).

    B = A Omega (Psi^T A Omega)^+ Psi^T A
    H_r^flat(A) = ||B||_F^2 + (1/(2r)) (||A G||_F^2 - ||B G||_F^2)

    The pseudoinverse uses jax.numpy.linalg.pinv with its default rcond.
    That rcond is a numerical choice the paper does not specify.

    Products: 4r columns with A (Omega and G; one batched call) and 4r columns
    with A.T. Non-adaptive. B and A are not materialized.

    Y = A Omega (m, 2r), Z = A.T Psi (n, 4r), AG = A G (m, 2r).
    C = Z.T @ Omega (4r, 2r), which is Psi.T A Omega. Z.T @ Y does not
    have matching inner dimensions when m is not n.
    M = pinv(C) (2r, 4r), P = Y @ M (m, 4r).
    ||B||_F^2 = trace((P.T @ P) @ (Z.T @ Z)).
    B G = Y @ M @ (Z.T @ G), and Z.T @ G equals Psi.T (A G).
    """
    if omega is not None and psi is not None and G is not None:
        omega = _as_float64(omega)
        psi = _as_float64(psi)
        G = _as_float64(G)
    else:
        k_omega, k_psi, k_G = _split_key(key, 3)
        if omega is None:
            omega = gaussian_sketch(k_omega, n, 2 * r)
        else:
            omega = _as_float64(omega)
        if psi is None:
            psi = gaussian_sketch(k_psi, m, 4 * r)
        else:
            psi = _as_float64(psi)
        if G is None:
            G = gaussian_sketch(k_G, n, 2 * r)
        else:
            G = _as_float64(G)

    # One batched product with A covering Omega and G (4r columns total).
    AG_and_omega = matvec(jnp.concatenate([omega, G], axis=1))
    Y = AG_and_omega[:, : 2 * r]
    AG = AG_and_omega[:, 2 * r :]
    Z = rmatvec(psi)
    # Psi.T @ A @ Omega = (A.T @ Psi).T @ Omega. Not Z.T @ Y.
    C = Z.T @ omega
    # Default rcond. The paper does not specify a cutoff.
    M = jax.numpy.linalg.pinv(C)
    P = Y @ M
    PtP = P.T @ P
    ZtZ = Z.T @ Z
    B_fro_sq = jnp.trace(PtP @ ZtZ)
    # B G = Y @ M @ (Psi.T @ A @ G) = Y @ M @ (Z.T @ G)
    BG = Y @ M @ (Z.T @ G)
    return B_fro_sq + (frobenius_sq(AG) - frobenius_sq(BG)) / (2.0 * r)


def hutch_flat_matrix(A, r, key=None, omega=None, psi=None, G=None):
    """Hutch-flat for an explicit array A."""
    A = _as_float64(A)
    matvec, rmatvec = matvec_rmatvec_from_matrix(A)
    m, n = A.shape
    return hutch_flat(
        matvec, rmatvec, m, n, r, key=key, omega=omega, psi=psi, G=G
    )


def hutch_flat_explicit(A, omega, psi, G):
    """Reference Hutch-flat that forms B densely via pinv. For tests only."""
    A = _as_float64(A)
    omega = _as_float64(omega)
    psi = _as_float64(psi)
    G = _as_float64(G)
    Y = A @ omega
    C = psi.T @ Y
    M = jax.numpy.linalg.pinv(C)
    B = Y @ M @ (psi.T @ A)
    AG = A @ G
    BG = B @ G
    r = omega.shape[1] // 2
    return frobenius_sq(B) + (frobenius_sq(AG) - frobenius_sq(BG)) / (2.0 * r)


def variance_hutch_sharp(frobenius_sq_value, schatten_4_fourth_value, r):
    """Closed form Var(H_r#) = (2/r^2) ||A||_(4)^4 + (2/r^2) ||A||_F^4.

    Theorem 1.1 assumes r >= 16. This function does not reject smaller r.
    ||A||_F^4 means (||A||_F^2)^2.
    """
    fro4 = frobenius_fourth_from_sq(frobenius_sq_value)
    sch4 = _as_float64(schatten_4_fourth_value)
    r = jnp.asarray(r, dtype=jnp.float64)
    return (2.0 / (r * r)) * sch4 + (2.0 / (r * r)) * fro4


def variance_weighted(frobenius_sq_value, schatten_4_fourth_value, r, alpha):
    """Var(H_{r,alpha}#) from the weighted closed form.

    ((1-alpha)^2 / r) ||A||_(4)^4 + (2 alpha^2 / r^2) (||A||_(4)^4 + ||A||_F^4).
    Theorem 1.1 assumes r >= 16. This function does not reject smaller r.
    """
    fro4 = frobenius_fourth_from_sq(frobenius_sq_value)
    sch4 = _as_float64(schatten_4_fourth_value)
    alpha = _as_float64(alpha)
    r = jnp.asarray(r, dtype=jnp.float64)
    return ((1.0 - alpha) ** 2) / r * sch4 + (2.0 * alpha ** 2) / (r * r) * (
        sch4 + fro4
    )
