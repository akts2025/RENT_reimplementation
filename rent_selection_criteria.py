import numpy as np
from scipy.stats import t

def coef_nonzero (coef_list):
    #percentage of iterations with non-zero coef
    result = np.count_nonzero(coef_list)/len(coef_list)
    return result

def coef_signstable (coef_list):
    #coefs have stable signs across iterations
    result = abs(sum(np.sign(coef_list)))/len(coef_list)
    return result

def coef_ttest (coef_list):
    #t-test for coef=0 across iterations
    means = np.mean(coef_list, axis=0)
    stds = np.std(coef_list, axis=0)
    ttest_result = t.cdf(abs(means / np.sqrt((stds ** 2) / len(coef_list))), (len(coef_list)-1))
    return ttest_result