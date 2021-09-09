from functools import partial
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

import statsmodels.api as sm
from bhhh import minimize_bhhh


def _cdf(X):
    return 1 / (1 + np.exp(-X))


def scoreobs(endog, exog, params):
    L = _cdf(np.dot(exog, params))
    return (endog - L)[:, None] * exog


def loglikeobs(endog, exog, params):
    q = 2 * endog - 1
    return np.log(_cdf(q * np.dot(exog, params)))


@pytest.fixture
def data():
    np.random.seed(12)

    num_observations = 5000
    x1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observations)

    endog = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

    simulated_exog = np.vstack((x1, x2)).astype(np.float32)
    exog = simulated_exog
    intercept = np.ones((exog.shape[0], 1))
    exog = np.hstack((intercept, exog))

    return endog, exog


def test_fail(data):
    endog, exog = data
    scoreobs_p = partial(scoreobs, endog, exog)
    loglikeobs_p = partial(loglikeobs, endog, exog)

    def logit_criterion(params):
        """Logit criterion function.
    
        Args:
            params (np.ndarray): Parameter vector of length n_obs.
    
        Returns:
            loglike (np.ndarray): Array of negative loglikelihood contributions of
                shape (n_obs,).
            jac (np.ndarray): Array of the jacobian of shape (n_obs, n_params).
        """
        jac = scoreobs_p(params)
        loglike = -loglikeobs_p(params)

        return loglike, jac

    params = np.zeros(exog.shape[1])
    calculated = minimize_bhhh(logit_criterion, params)

    sm_res = sm.Logit(endog, exog).fit()
    expected = sm_res.params

    aaae(calculated, expected, decimal=4)
