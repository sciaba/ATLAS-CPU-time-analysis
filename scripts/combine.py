#! /bin/env python

import math
import csv
from copy import deepcopy
from collections import defaultdict

import numpy as np

no_files = 10
input_pattern = 'results_cputype_task_cpu_sub_%s_pile_0.001.csv'
output_file = 'results_cputype_task_cpu_pile_0.001.csv'
#input_pattern = 'results_cputype_task_cpu_sub2_%s_hc.csv'
#output_file = 'results_cputype_task_cpu_hcx.csv'

def rescale_1norm(x, x_err):   # rescales x such that its l-1 norm is one 
    n = len(x) - np.isnan(x).sum()
    s = np.nansum(np.fabs(x))
    return x / s * n, x_err / s * n

def rescale_dist(x, y, y_err):
    y2 = y * x / x
    alpha = np.nansum(x * y) / np.nansum(y2 * y2)
    return y * alpha, y_err * alpha

def rescale_first(x, y, y_err):
    return y / y[0] * x[0], y_err / y[0] * x[0]

def nanaverage(a, weights):
    average = np.nansum(a / weights**2) / np.nansum(1 / weights**2)
    return average

factors = defaultdict(dict)
factors_err = defaultdict(dict)
for i in range(no_files):
    with open(input_pattern % (str(i))) as f:
        r = csv.reader(f, delimiter=',')
        for row in r:
            try:
                (cpu, k, k_err) = row
                factors[cpu][i] = float(k)
                factors_err[cpu][i] = float(k_err)
            except:
                try:
                    (cpu, k) = row
                    factors[cpu][i] = float(k)
                    factors_err[cpu][i] = 1e-4
                except:
                    print row

cmap = dict()
i = -1
fact_all = np.zeros((len(factors), no_files))
fact_err_all = np.zeros((len(factors), no_files))
for c in factors.keys():
    i += 1
    cmap[c] = i
    fact_all[i] = [factors[c].setdefault(f, np.nan) for f in range(no_files)]
    fact_err_all[i] = [factors_err[c].setdefault(f, np.nan) for f in range(no_files)]
fact_all_t = fact_all.transpose()
fact_err_all_t = fact_err_all.transpose()

fact_all_resc_t = np.empty_like(fact_all_t)
fact_err_all_resc_t = np.empty_like(fact_err_all_t)
for i in range(no_files):
    v = fact_all_t[i]
    v_e = fact_err_all_t[i]
    (v1, v1_e) = rescale_1norm(v, v_e)
    fact_all_resc_t[i] = deepcopy(v1)
    fact_err_all_resc_t[i] = deepcopy(v1_e)

fact_all_resc = fact_all_resc_t.transpose()
fact_err_all_resc = fact_err_all_resc_t.transpose()

z = list()
with open(output_file, 'wb') as csvfile:
    w = csv.writer(csvfile, delimiter=',')
    for c in sorted(cmap.keys()):
        w_average = nanaverage(fact_all_resc[cmap[c]], fact_err_all_resc[cmap[c]])
        w_average_err = math.sqrt(1 / np.nansum(1 / fact_err_all_resc[cmap[c]]**2))
        w_average_spread = np.nanstd(fact_all_resc[cmap[c]]) / math.sqrt(no_files)
#        print '%.4f\t%.4f\t%.4f\t%.4f' % (w_average, w_average_err, w_average_spread,
#                                          w_average_spread / w_average_err)
        z.append(w_average_spread / w_average_err)
        w.writerow([c, w_average, w_average_err])

scale = np.nanmean(z)
print 'Errors calculated in the fit are underestimated by %.1f' % scale
