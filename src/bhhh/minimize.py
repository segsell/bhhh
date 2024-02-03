"""Implementation of the unconstrained Berndt-Hall-Hall-Hausman (BHHH) algorithm."""
from typing import Optional

import numpy as np


def minimize_bhhh(
    criterion: callable,
    derivative: callable,
    x: np.ndarray,
    counts=None,
    convergence_absolute_gradient_tolerance: Optional[float] = 1e-8,
    stopping_max_iterations: Optional[int] = 200,
) -> np.ndarray:
    """Minimize a likelihood function using the BHHH algorithm.

    The BHHH algorithm is an iterative method to minimize a likelihood function. It
    uses the Hessian approximation to calculate the direction of the next step. The
    algorithm is based on the Newton-Raphson method and is used to find the maximum
    likelihood estimates of the parameters.

    Note that this function expects a criterion function that returns the negative
    loglikelihood contributions of the model and a derivative function that returns
    the scores of the model. Finding the minimum of the negative loglikelihood is
    equivalent to finding the maximum of the likelihood.

    The internal criterion_candidate vector of shape (n_obs,) contains the likelihood
    contributions of the model. The jacobian matrix of shape (n_obs, n_params)
    contains the scores of the model.

    Args:
        criterion (callable): The objective function to be minimized, i.e. the
            negative loglikelihood contributions of the model.
        derivative (callable): Function returning the derivative of the
            objective function, i.e. the scores of the model.
        x (np.ndarray): Initial guess of the parameter vector (starting points)
            of shape (n_params,).
        convergence_absolute_gradient_tolerance (float): Stopping criterion for the
            gradient tolerance.
        stopping_max_iterations (int): Maximum number of iterations. If reached,
            terminate.

    Returns:
        (dict) Result dictionary containing:

        - solution_x (np.ndarray): Solution vector of shape (n_params,).
        - solution_criterion (np.ndarray): Array of shape (n_obs,) containing the
            likelihood contributions at the solution. The sum of this array is the
            negative loglikelihood at the solution.
        - n_iterations (int): Number of iterations the algorithm ran before finding a
            solution or reaching stopping_max_iterations.

    """
    # Define internal functions to handle cell based likelihood:
    if counts is None:

        def criterion_internal(x):
            return criterion(x)

        def derivative_internal(x):
            return derivative(x)

        def proxy_hessian(scores):
            return np.dot(scores.T, scores)

    x_accepted = x

    criterion_accepted = criterion_internal(x)
    score = derivative_internal(x)

    hessian_approx = np.dot(score.T, score)
    jacobian = np.sum(score, axis=0)
    direction = np.linalg.solve(hessian_approx, jacobian)
    gtol = np.dot(jacobian, direction)

    initial_step_size = 1
    step_size = initial_step_size
    last_step_accepted = True

    niter = 0
    while niter < stopping_max_iterations:
        niter += 1

        x_candidate = x_accepted - step_size * direction
        criterion_candidate = criterion_internal(x_candidate)

        # If previous step was accepted
        if last_step_accepted:
            score = derivative_internal(x_candidate)
            hessian_approx = np.dot(score.T, score)

        # Line search
        if np.sum(criterion_candidate) > np.sum(criterion_accepted):
            last_step_accepted = False
            step_size /= 2

            if step_size <= 0.01:
                # Accept step
                x_accepted = x_candidate
                criterion_accepted = criterion_candidate

                # Reset step size
                step_size = initial_step_size
                last_step_accepted = True

        # If decrease in likelihood, calculate new direction vector
        else:
            # Accept step
            x_accepted = x_candidate
            criterion_accepted = criterion_candidate

            score = derivative_internal(x_accepted)
            jacobian = np.sum(score, axis=0)
            direction = np.linalg.solve(hessian_approx, jacobian)
            gtol = np.dot(jacobian, direction)

            if gtol > 0:
                hessian_approx = np.dot(score.T, score)
                direction = np.linalg.solve(hessian_approx, jacobian)

            # Reset stepsize
            step_size = initial_step_size
            last_step_accepted = True

        if abs(gtol) < convergence_absolute_gradient_tolerance:
            break

    result_dict = {
        "solution_x": x_accepted,
        "solution_criterion": criterion_accepted,
        "n_iterations": niter,
    }

    return result_dict
