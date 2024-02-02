"""Implementation of the unconstrained Berndt-Hall-Hall-Hausman (BHHH) algorithm."""
from typing import Optional

import numpy as np


def minimize_bhhh(
    criterion_and_derivative: callable,
    x: np.ndarray,
    convergence_absolute_gradient_tolerance: Optional[float] = 1e-8,
    stopping_max_iterations: Optional[int] = 200,
) -> np.ndarray:
    """Minimize a likelihood function using the BHHH algorithm.

    Args:
        criterion_and_derivative (callable): The objective function to be minimized.
        x (np.ndarray): Initial guess of the parameter vector (starting points).
        convergence_absolute_gradient_tolerance (float): Stopping criterion for the
            gradient tolerance.
        stopping_max_iterations (int): Maximum number of iterations. If reached,
            terminate.

    Returns:
        (dict) Result dictionary containing:

        - solution_x (np.ndarray): Solution vector of shape (n,).
        - solution_criterion (np.ndarray): Likelihood at the solution. Shape (n_obs,).
        - n_iterations (int): Number of iterations the algorithm ran before finding a
            solution or reaching stopping_max_iterations.

    """
    criterion_accepted, gradient = criterion_and_derivative(x)
    x_accepted = x

    hessian_approx = np.dot(gradient.T, gradient)
    gradient_sum = np.sum(gradient, axis=0)
    direction = np.linalg.solve(hessian_approx, gradient_sum)
    gtol = np.dot(gradient_sum, direction)

    initial_step_size = 1
    step_size = initial_step_size

    niter = 1
    while niter < stopping_max_iterations:
        niter += 1

        x_candidate = x_accepted + step_size * direction
        criterion_candidate, gradient = criterion_and_derivative(x_candidate)

        # If previous step was accepted
        if step_size == initial_step_size:
            hessian_approx = np.dot(gradient.T, gradient)

        else:
            criterion_candidate, gradient = criterion_and_derivative(x_candidate)

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

            gradient_sum = np.sum(gradient, axis=0)
            direction = np.linalg.solve(hessian_approx, gradient_sum)
            gtol = np.dot(gradient_sum, direction)

            if gtol < 0:
                hessian_approx = np.dot(gradient.T, gradient)
                direction = np.linalg.solve(hessian_approx, gradient_sum)

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
