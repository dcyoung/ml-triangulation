import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
from uncertainty import ECELoss


def get_calibrated_temperature(logits: npt.NDArray, labels: npt.NDArray) -> float:
    ece_criterion = ECELoss()

    def ece_eval(temperature):
        loss = ece_criterion.loss(logits / temperature, labels, 15)
        return loss

    temperature_for_min, min_value, _ = opt.fmin_l_bfgs_b(
        ece_eval, np.array([1.0]), approx_grad=True, bounds=[(0.001, 100)]
    )
    temperature_for_min = temperature_for_min[0]
    return temperature_for_min
