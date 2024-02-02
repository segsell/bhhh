from typing import Optional

import numpy as np


def minimize_bhhh(
    fun: callable,
    x0: np.ndarray,
    tol: Optional[float] = 1e-8,
    maxiter: Optional[int] = 100,
) -> np.ndarray:
    """Minimization of a scalar function via the BHHH algorithm.

    Args:
        fun (callable): The objective function to be minimized.
        x0 (np.ndarray): Initial guess. Array of real elements of size (n,),
            where `n` is the number of independent variables, i.e. parameters.
        tol (float): Tolerance for termination.
        maxiter(int): Maximum number of iterations to perform.

    Returns:
        x_hat(np.ndarray): The solution array of size (n,) containing fitted
        parameter values.
    """
    old_fval, old_jac = fun(x0, task="criterion_and_derivative")

    # Approxmiate Hessian as the outer product of the Jacobian
    hess_approx = np.dot(old_jac.T, old_jac)

    jac_sum = np.sum(hess_approx, axis=0)
    direc = np.linalg.solve(hess_approx, jac_sum)  # Current direction set
    jacdirec = np.dot(jac_sum, direc)

    # Initialize step size
    lambda0 = 1
    lambdak = lambda0

    for _ in range(maxiter):
        xk = x0 + lambdak * direc

        fval = fun(xk, task="criterion")

        # If previous step was accepted
        if lambdak == lambda0:
            jac = fun(xk, task="derivative")
            hess_approx = np.dot(jac.T, jac)

        # Line search
        if np.sum(fval) > np.sum(old_fval):
            lambdak /= 2

            if lambdak <= 0.01:
                # Accept step
                x0 = xk
                old_fval = fval

                # Reset step size
                lambdak = lambda0

        # If decrease in likelihood, calculate new direction vector
        else:
            # Accept step
            x0 = xk
            old_fval = fval

            jac = fun(xk, task="derivative")
            jac_sum = np.sum(jac, axis=0)
            direc = np.linalg.solve(hess_approx, jac_sum)
            jacdirec = np.dot(jac_sum, direc)

            # Reset stepsize
            lambdak = lambda0

        if jacdirec < tol:  # Stopping rule
            break

    x_hat = x0

    return x_hat
