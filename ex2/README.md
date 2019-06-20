# Exercise 2 - Logistic Regression
- similar to linear regression but uses a different *hypothesis* in order to
**classify** objects into 2 classes;
- the vectorisation effort is still ongoing.

## Unregularised Logistic Regression
- `h` now is `g(X * theta)`, which requires the usage of `log` in order keep `J`
convex;
- ther than that, `J` has the same meaning as in linear regression;
- in order to use an algorithm faster than *Gradient Descent*, the *Octave*
function, `fminunc` is called, which uses the [Conjugate Gradient](https://en.wikipedia.org/wiki/Conjugate_gradient_method),
or [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS);
- hence the need to provide the gradient of `J` as well.

## Regularised Logistic Regression
- now the `costFunctionReg` takes on the return value of the previous `costFunction`
and applies regularisation by trying to minimise `theta_j`, `j >= 1`;
- the rest of the algorithm is identical to the previous, unregularised
implementation.
