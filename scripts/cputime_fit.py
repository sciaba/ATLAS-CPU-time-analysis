#! /usr/bin/env python

import sys
import csv
import getopt
from collections import defaultdict, OrderedDict, Counter
from copy import copy
import pprint as pp
import numpy as np
import pandas as pd
import scipy.optimize as optimize

# Input parameters
input_file = sys.argv[1]
event_type = sys.argv[2]
site_wc_contrib = sys.argv[3]
mode = 'cputype'
output_file = 'results_' + mode + '_' + input_file + '_' + event_type + '_' + site_wc_contrib + '.csv'

df = pd.read_csv(input_file + '.csv',
                 names=['jeditaskid',
                        'processingtype',
                        'transformation',
                        'atlrel',
                        'site',
                        'cputype',
                        'njobs',
                        'cpuevt_avg',
                        'cpuevt_rms',
                        'wallevt_avg',
                        'wallevt_rms',
                        'cpu',
                        'wc',
                        'cores',
                        'jevts',
                        'eff'],
                 )

# Find sites contributing more than a given fraction to the total wall-clock
grouped = df[df.processingtype == event_type].groupby([mode], as_index=False)
tot_wc = grouped['wc'].sum()
a = tot_wc.sort_values('wc', ascending=False)
bigtotal_wc = a.wc.sum()
big_sites = a[a.wc > bigtotal_wc * float(site_wc_contrib)][mode].tolist()

site_ref = big_sites[0]

grouped = df.groupby(['processingtype', 'jeditaskid', mode], as_index=False, sort=False)

data = defaultdict(OrderedDict)
for (p, j, s), g in grouped:
    if s not in big_sites: continue
    if p != event_type: continue
    x = np.average(g.cpuevt_avg, weights=g.wc)
    data[j][s] = x

# Remove from data tasks with only one site
for (k, v) in data.items():
    if len(v) == 1:
        del data[k]

# Replace site names with numbers
i = -1
sites = set()
smap = dict()
for (k, v) in data.items():
    for s in v.keys():
        if s not in sites:
            i += 1
            smap[s] = i
            sites.add(s)
        data[k][smap[s]]= data[k][s]
        del data[k][s]

nsites = len(smap)
print len(sites)
print 'There are %s sites in the fit' % nsites

k_ini = np.ones(nsites)   # Initialises CPU factors for all sites to one

# The function to minimise is the sum of the squares of the differences
# between the ratio of the average cpu times per event at two sites
# running the same task and the ratio of the corresponding k factors.
# For each task, the sites are ordered by increasing site number and
# the ratios between one site and the next one are considered.
def func(k):
    total = 0.
    grad = np.zeros(len(k))
    for _, task in data.items():
        if len(task) == 1: continue
        ns = np.array(task.keys())
        kf = np.array([k[i] for i in task.keys()])
        v = np.array(task.values())
        kf2 = np.roll(kf, -1)
        v2 = np.roll(v, -1)
        ratio = (v2 / v - kf / kf2)**2
        ratio[np.isnan(ratio)] = 0.
        ratio[np.isinf(ratio)] = 0.
        delta = np.sum(ratio[:-1])
        total += delta
        for i in range(len(task)):
            if i == len(task) - 1:
                x1 = 0.
            else:
                x1 = -2 * (v[i+1] / v[i] - kf[i] / kf[i+1]) / kf[i+1]
            if np.isnan(x1) or np.isinf(x1): x1 = 0.
            if i == 0:
                x2 = 0.
            else:
                x2 = 2 * (v[i] / v[i-1] - kf[i-1] / kf[i]) * kf[i-1] / kf[i]**2
            if np.isnan(x2) or np.isinf(x2): x2 = 0.
            grad[ns[i]] += x1 + x2
    return total, grad

def func2(k):
    total = 0.
    for _, task in data.items():
        kf = np.array([k[i] for i in task.keys()])
        v = np.array(task.values())
        kf2 = np.roll(kf, -1)
        v2 = np.roll(v, -1)
        ratio = (v2 / v - kf / kf2)**2
        ratio[np.isnan(ratio)] = 0.
        ratio[np.isinf(ratio)] = 0.
        delta = np.sum(ratio[:-1])
        total += delta
    return total

def grad(k):
    grad = np.zeros(len(k))
    for _, task in data.items():
        ns = np.array(task.keys())
        kf = np.array([k[i] for i in task.keys()])
        v = np.array(task.values())
        for i in range(len(task)):
            if i == len(task) - 1:
                x1 = 0.
            else:
                x1 = -2 * (v[i+1] / v[i] - kf[i] / kf[i+1]) / kf[i+1]
            if np.isnan(x1) or np.isinf(x1): x1 = 0.
            if i == 0:
                x2 = 0.
            else:
                x2 = 2 * (v[i] / v[i-1] - kf[i-1] / kf[i]) * kf[i-1] / kf[i]**2
            if np.isnan(x2) or np.isinf(x2): x2 = 0.
            grad[ns[i]] += (x1 + x2)
    return grad

print 'Initial function value: ', func2(k_ini)
check = optimize.check_grad(func2, grad, k_ini)
print check

# Conventionally, use CERN-PROD as reference (k-factor is 1.)
#ref = smap['CERN-PROD']
ref = smap[site_ref]
cons = ({'type': 'eq', 'fun': lambda x: x[ref] - 1})

# Execute fit
result = optimize.minimize(func2, k_ini, method='SLSQP', jac=grad, constraints=cons, options={'disp': True, 'maxiter': 2000})
print result
k = result.x

for s in sorted(smap.keys()):
    print '%s,%f' % (s, k[smap[s]])

# Write output to file
with open(output_file, 'wb') as csvfile:
    w = csv.writer(csvfile, delimiter=',')
    for s in sorted(smap.keys()):
        w.writerow([s, k[smap[s]]])

