"""Optional test using the ruspy package."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bhhh.minimize import minimize_bhhh
from ruspy.estimation.criterion_function import get_criterion_function

TEST_DIR = Path(__file__).parent


@pytest.fixture()
def input_data():
    df_in = pd.read_pickle(TEST_DIR / "replication_ruspy" / "group_4.pkl")
    return df_in


@pytest.fixture()
def init_dict():
    return {
        "model_specifications": {
            "discount_factor": 0.9999,
            "num_states": 90,
            "maint_cost_func": "linear",
            "cost_scale": 1e-3,
        },
        "method": "NFXP_BHHH",
    }


def test_get_criterion_function(input_data, init_dict):
    func_dict, _ = get_criterion_function(init_dict, input_data)
    result_bhhh = minimize_bhhh(
        criterion=func_dict["criterion_function"],
        derivative=func_dict["criterion_derivative"],
        x=np.array([10, 2]),
        convergence_absolute_gradient_tolerance=1e-12,
        stopping_max_iterations=500,
    )
    sol = np.array([10.0749422, 2.29309298])
    np.testing.assert_allclose(result_bhhh["solution_x"], sol, atol=1e-5)
    np.testing.assert_allclose(result_bhhh["solution_criterion"].sum(), 163.58428365)
    gradient = func_dict["criterion_derivative"](result_bhhh["solution_x"])
    np.testing.assert_allclose(gradient.sum(), 0, atol=1e-4)
