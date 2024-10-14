"""Implementation of the unconstrained Berndt-Hall-Hall-Hausman (BHHH) algorithm."""

from typing import Optional

import numpy as np


def minimize_bhhh(
    criterion: callable,
    derivative: callable,
    x: np.ndarray,
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
    x_accepted = x

    criterion_accepted = criterion(x)
    jacobian = derivative(x)

    hessian_approx = np.dot(jacobian.T, jacobian)
    gradient = np.sum(jacobian, axis=0)
    direction = np.linalg.solve(hessian_approx, gradient)
    gtol = np.dot(gradient, direction)

    initial_step_size = 1
    step_size = initial_step_size

    niter = 1
    while niter < stopping_max_iterations:
        niter += 1

        x_candidate = x_accepted + step_size * direction
        criterion_candidate = criterion(x_candidate)

        # If previous step was accepted
        if step_size == initial_step_size:
            jacobian = derivative(x_candidate)
            hessian_approx = np.dot(jacobian.T, jacobian)

        # Line search
        if np.sum(criterion_candidate) > np.sum(criterion_accepted):
            step_size /= 2

            if step_size <= 0.01:
                # Accept step
                x_accepted = x_candidate
                criterion_accepted = criterion_candidate

                # Reset step size
                step_size = initial_step_size

        # If decrease in likelihood, calculate new direction vector
        else:
            # Accept step
            x_accepted = x_candidate
            criterion_accepted = criterion_candidate

            jacobian = derivative(x_accepted)
            gradient = np.sum(jacobian, axis=0)
            direction = np.linalg.solve(hessian_approx, gradient)
            gtol = np.dot(gradient, direction)

            if gtol < 0:
                hessian_approx = np.dot(jacobian.T, jacobian)
                direction = np.linalg.solve(hessian_approx, gradient)

            # Reset stepsize
            step_size = initial_step_size

        if gtol < convergence_absolute_gradient_tolerance:
            break

    result_dict = {
        "solution_x": x_accepted,
        "solution_criterion": criterion_accepted,
        "n_iterations": niter,
    }

    return result_dict
