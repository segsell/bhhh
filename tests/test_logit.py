"""Test the BHHH algorithm against statsmodels."""

import numpy as np
import pytest
import statsmodels.api as sm
from numpy.testing import assert_array_almost_equal as aaae
from scipy.stats import norm

from bhhh.minimize import minimize_bhhh

# =====================================================================================
# Test Data
# =====================================================================================


def generate_test_data(seed=12):
    rng = np.random.default_rng(seed)

    num_observations = 5000
    x1 = rng.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observations)
    x2 = rng.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observations)

    endog = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

    simulated_exog = np.vstack((x1, x2)).astype(np.float32)
    exog = simulated_exog
    intercept = np.ones((exog.shape[0], 1))
    exog = np.hstack((intercept, exog))

    return endog, exog


@pytest.fixture()
def result_statsmodels_logit():
    endog, exog = generate_test_data()
    result = sm.Logit(endog, exog).fit()

    return result


@pytest.fixture()
def result_statsmodels_probit():
    endog, exog = generate_test_data()
    result = sm.Probit(endog, exog).fit()

    return result


# =====================================================================================
# Logit
# =====================================================================================


def _cdf_logit(x):
    return 1 / (1 + np.exp(-x))


def get_loglikelihood_logit(endog, exog, x):
    q = 2 * endog - 1
    linear_prediction = np.dot(exog, x)

    return np.log(_cdf_logit(q * linear_prediction))


def get_score_logit(endog, exog, x):
    linear_prediction = np.dot(exog, x)

    return (endog - _cdf_logit(linear_prediction))[:, None] * exog


def criterion_logit(x):
    """Return Logit criterion.

    Args:
        x (np.ndarray): Parameter vector of shape (n_params,).

    Returns:
        np.ndarray: 2d array of shape (n_obs,) containing the negative
            loglikelihood contributions of the Logit model.

    """
    endog, exog = generate_test_data()
    return -get_loglikelihood_logit(endog=endog, exog=exog, x=x)


def derivative_logit(x):
    """Return Logit derivative.

    Args:
        x (np.ndarray): Parameter vector of shape (n_params,).

    Returns:
        np.ndarray: 2d array of shape (n_obs, n_params) containing
            of shape (n_obs, n_params).

    """
    endog, exog = generate_test_data()
    return -get_score_logit(endog=endog, exog=exog, x=x)


# =====================================================================================
# Probit
# =====================================================================================


def get_loglikelihood_probit(endog, exog, x):
    q = 2 * endog - 1
    linear_prediction = np.dot(exog, x[: exog.shape[1]])

    return np.log(norm.cdf(q * linear_prediction))


def get_score_probit(endog, exog, x):
    q = 2 * endog - 1
    linear_prediction = np.dot(exog, x[: exog.shape[1]])

    derivative_loglikelihood = (
        q * norm.pdf(q * linear_prediction) / norm.cdf(q * linear_prediction)
    )

    return derivative_loglikelihood[:, None] * exog


def criterion_probit(x):
    """Return Probit criterion.

    Args:
        x (np.ndarray): Parameter vector of shape (n_params,).

    Returns:
        np.ndarray: 2d array of shape (n_obs,) containing the negative
            loglikelihood contributions of the Probit model.

    """
    endog, exog = generate_test_data()
    return -get_loglikelihood_probit(endog=endog, exog=exog, x=x)


def derivative_probit(x):
    """Return Probit derivative.

    Args:
        x (np.ndarray): Parameter vector of shape (n_params,).

    Returns:
        np.ndarray: 2d array of shape (n_obs, n_params) containing
            the scores of the Probit model.

    """
    endog, exog = generate_test_data()
    return -get_score_probit(endog=endog, exog=exog, x=x)


# =====================================================================================
# Tests
# =====================================================================================


@pytest.mark.parametrize(
    ("criterion", "derivative", "result_statsmodels"),
    [
        (criterion_logit, derivative_logit, "result_statsmodels_logit"),
        (criterion_probit, derivative_probit, "result_statsmodels_probit"),
    ],
)
def test_maximum_likelihood(criterion, derivative, result_statsmodels, request):
    result_expected = request.getfixturevalue(result_statsmodels)

    x = np.zeros(3)

    result_bhhh = minimize_bhhh(
        criterion=criterion,
        derivative=derivative,
        x=x,
        convergence_absolute_gradient_tolerance=1e-8,
        stopping_max_iterations=200,
    )

    aaae(result_bhhh["solution_x"], result_expected.params, decimal=4)


@pytest.mark.parametrize(
    ("criterion", "derivative", "result_statsmodels"),
    [
        (criterion_logit, derivative_logit, "result_statsmodels_logit"),
        (criterion_probit, derivative_probit, "result_statsmodels_probit"),
    ],
)
def test_maximum_likelihood_cell_based(
    criterion, derivative, result_statsmodels, request
):
    result_expected = request.getfixturevalue(result_statsmodels)

    x = np.zeros(3)
    n_obs = criterion(x).shape[0]

    result_bhhh = minimize_bhhh(
        criterion=criterion,
        derivative=derivative,
        x=x,
        convergence_absolute_gradient_tolerance=1e-12,
        stopping_max_iterations=200,
        counts=np.ones(n_obs),
    )

    aaae(result_bhhh["solution_x"], result_expected.params, decimal=4)


@pytest.mark.parametrize(
    ("criterion", "derivative", "result_statsmodels"),
    [
        (criterion_logit, derivative_logit, "result_statsmodels_logit"),
        (criterion_probit, derivative_probit, "result_statsmodels_probit"),
    ],
)
def test_maximum_likelihood_cell_based_aux(
    criterion, derivative, result_statsmodels, request
):
    result_expected = request.getfixturevalue(result_statsmodels)

    x = np.zeros(3)
    n_obs = criterion(x).shape[0]

    def crit_with_aux(x, aux):
        return criterion(x), aux

    def deriv_with_aux(x, aux):
        return derivative(x), aux

    result_bhhh = minimize_bhhh(
        criterion=crit_with_aux,
        derivative=deriv_with_aux,
        x=x,
        convergence_absolute_gradient_tolerance=1e-12,
        stopping_max_iterations=200,
        counts=np.ones(n_obs),
        aux_data=1,
    )

    aaae(result_bhhh["solution_x"], result_expected.params, decimal=4)


@pytest.mark.parametrize(
    ("criterion", "derivative", "result_statsmodels"),
    [
        (criterion_logit, derivative_logit, "result_statsmodels_logit"),
        (criterion_probit, derivative_probit, "result_statsmodels_probit"),
    ],
)
def test_maximum_likelihood_aux(criterion, derivative, result_statsmodels, request):
    result_expected = request.getfixturevalue(result_statsmodels)

    x = np.zeros(3)

    def crit_with_aux(x, aux):
        return criterion(x), aux

    def deriv_with_aux(x, aux):
        return derivative(x), aux

    result_bhhh = minimize_bhhh(
        criterion=crit_with_aux,
        derivative=deriv_with_aux,
        x=x,
        convergence_absolute_gradient_tolerance=1e-12,
        stopping_max_iterations=200,
        aux_data=1,
    )

    aaae(result_bhhh["solution_x"], result_expected.params, decimal=4)
