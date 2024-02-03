"""Implementation of the unconstrained Berndt-Hall-Hall-Hausman (BHHH) algorithm."""
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np


def minimize_bhhh(
    criterion: callable,
    derivative: callable,
    x: np.ndarray,
    counts: np.ndarray = None,
    has_aux: bool = False,
    aux_start: Optional[np.ndarray] = None,
    convergence_absolute_gradient_tolerance: Optional[float] = 1e-8,
    stopping_max_iterations: Optional[int] = 200,
) -> Dict[str, Union[np.ndarray, int]]:
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
        counts (np.ndarray): Array of shape (n_obs,) containing the number of
            observations for each likelihood contribution. If None, the criterion
            function is assumed to return for each observation one contribution.
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
    if has_aux and aux_start is None:
        raise ValueError("Auxiliary inputs are required but not provided.")

    criterion_internal, derivative_internal, proxy_hessian = process_functions(
        criterion, derivative, counts, has_aux
    )

    x_accepted = x

    criterion_accepted, aux = criterion_internal(x, aux_start)
    score, aux = derivative_internal(x, aux)

    hessian_approx = proxy_hessian(score)
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
        criterion_candidate, aux = criterion_internal(x_candidate, aux)

        # If previous step was accepted
        if last_step_accepted:
            score, aux = derivative_internal(x_candidate, aux)
            hessian_approx = proxy_hessian(score)

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

            score, aux = derivative_internal(x_accepted, aux)
            jacobian = np.sum(score, axis=0)
            direction = np.linalg.solve(hessian_approx, jacobian)
            gtol = np.dot(jacobian, direction)

            if gtol > 0:
                hessian_approx = proxy_hessian(score)
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


def process_functions(criterion, derivative, counts, has_aux):
    """This function process the criterion and derivative function, such that they
    can handle auxilary data as well as a cell based likelihood. If has_aux is True
    criterion and derivative function ar both expected to take two arguments and
    return two arguments."""
    if has_aux:

        def criterion_internal_aux(x, aux):
            return criterion(x, aux)

        def derivative_internal_aux(x, aux):
            return derivative(x, aux)

    else:

        def criterion_internal_aux(x, aux):
            return criterion(x), None

        def derivative_internal_aux(x, aux):
            return derivative(x), None

    if counts is None:

        def criterion_internal(x, aux):
            return criterion_internal_aux(x, aux)

        def derivative_internal(x, aux):
            return derivative_internal_aux(x, aux)

        def proxy_hessian(score):
            return np.dot(score.T, score)

    else:
        n_obs = np.sum(counts)
        sqrt_counts = np.sqrt(counts.clip(min=1))

        def criterion_internal(x, aux):
            return criterion_internal_aux(x, aux) * (counts / n_obs)

        def derivative_internal(x, aux):
            return derivative_internal_aux(x, aux) * (counts / n_obs)[:, None]

        def proxy_hessian(score):
            weighted_score = score / sqrt_counts[:, None]
            return np.dot(weighted_score.T, weighted_score) / n_obs

    return criterion_internal, derivative_internal, proxy_hessian
