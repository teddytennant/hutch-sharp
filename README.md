This is Hutch#, the non-adaptive Frobenius norm estimator of arXiv 2609.28472.

Install from this directory with `pip install -e .`. Run the demo with `hutch-sharp-demo`. Run the tests with `JAX_PLATFORMS=cpu python -m pytest -q`.

Callers pass `matvec(V)` for `V` of shape `(n, k)` and `rmatvec(U)` for `U` of shape `(m, k)`. Matrix wrappers accept an explicit array. Pass a pre-drawn Gaussian sketch or a PRNG key.

What does not match the paper: the demo uses n=64 and 40 trials, not Figure 1's n=1000 and 200 trials. Theorem 1.1's assumption r >= 16 is not enforced. The pseudoinverse rcond is JAX's default, which the paper does not specify. The S_hat == 0 edge case for alpha-hat (alpha-hat = 1 if both estimates are zero, and alpha-hat = 0 if S_hat is zero and F_hat is positive) is not in the paper. Median-of-means, Hutch++, and Nyström++ are not included. Sketches are Gaussian only.
