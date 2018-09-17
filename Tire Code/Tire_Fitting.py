import numpy as np
import scipy.optimize as optimization


def model(slip, coeff):
    force = coeff[2] * np.sin(coeff[1] * np.arctan(coeff[0] * (1 - coeff[3]) * (slip - coeff[5]) + coeff[3] *
                                                   np.arctan(coeff[0] * (slip - coeff[5])))) + coeff[4]
    return force


def residuals(coeff, force, slip):
    """returns the difference between the Pacejka model prediction and the actualy data force point
    do not square this! The scipy.least_square() function does it for us already!!"""
    residual = (force - model(slip, coeff))
    return residual


def least_square(slip_list, force_list):
    """fits the Pacejka equation to a pair of data(slip, force)
    EQUATION:F = D * SIN(C * ARCTAN(B(SLIP) - E * (B(SLIP) - ARCTAN(SLIP)))) + V """
    # don't forget Slip[tire][slip_angle_sign][psi][camber][load][instance] same for fy and mx
    # coefficients[tire][slip_angle_sign][psi][camber][load][individual coefficient]
    slip_list = np.array(slip_list)
    force_list = np.array(force_list)
    # updating the guess to go faster and make the fit more accurate. T plus correspond to + slip angle
    # be careful with bounds especially the last 2 as having loose bounds will let it translate along the axis too much
    if len(slip_list) > 50:
        max_val = abs((max(min(force_list), max(force_list), key=abs)))
        y_shift = .5 * max_val
        # print 'max_val', max_val, -max_val
        # print'y_shift', y_shift, -y_shift

        # x0 is initial guess
        x0 = np.array([.01, .01, max_val, .02, 0, 0], dtype=float)

        # the solver from scipy that makes all of this possible. It is some what sensitive to initial guess and bounds
        line = optimization.least_squares(residuals, x0, args=(force_list, slip_list), max_nfev=100,
                                          bounds=([-2, -3, -1.5*max_val, 0, -y_shift, -5],
                                                  [2, 3, 1.5 * max_val, 1, y_shift, 5]))

        return line.x
