import Tire_Fitting


def mean_calc(fy):
    total = 0
    # summing all entries of fy
    for i in range(len(fy)):
        total += fy[i]
    mean = total / (len(fy))
    print 'mean through'
    return mean


# i=tire/j=slip angle sign/k=psi/l=camber/m=load


def sum_square_residuals(force, sa, coeff_list):
    summ = 0
    for h in range(len(force)):
        summ += (force[h] - Tire_Fitting.model(sa[h], coeff_list)) ** 2
    print 'ssr through'
    return summ


def total_sum_squares(force, mean):
    summ = 0
    for i in range(len(force)):
        summ += (force[i] - mean) ** 2
    print 'tss through'
    return summ


def r_squared(force, sa, coefflist):
    ss_res = sum_square_residuals(force, sa, coefflist)
    mean = mean_calc(force)
    ss_tot = total_sum_squares(force, mean)
    # print'ss_res %s, ss_tot %s' % (ss_res, ss_tot)
    r2 = (1 - (ss_res / ss_tot))
    # print 'r2', r2
    return r2
