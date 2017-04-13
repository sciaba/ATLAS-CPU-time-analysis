#! /usr/bin/env python

import sys
import math
import csv
import argparse
import ntpath
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import scipy.optimize as optimize

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_file', help='data file to analyse')
parser.add_argument('event_type', help='job processing type to analyse')
parser.add_argument('wc_contrib',
                    help='lower limit on contribution of site/CPU '
                    'to the global wall-clock time', type=float)
parser.add_argument('mode',
                    help='"site" or "cputype" depending on the entity '
                    'with respect to which to perform the analysis')
args = parser.parse_args()

output_file = ('results_' + args.mode + '_' + ntpath.basename(args.data_file) 
               + '_' + args.event_type + '_' + str(args.wc_contrib) + '.csv')

df = pd.read_csv(args.data_file + '.csv',
                 names=['jeditaskid', 'processingtype', 'transformation',
                        'atlrel', 'site', 'cputype', 'njobs', 'cpuevt_avg',
                        'cpuevt_rms', 'wallevt_avg', 'wallevt_rms', 'cpu',
                        'wc', 'actualcores', 'cores', 'jevts','eff'])

# Find sites contributing more than a given fraction to the total wall-clock
grouped = df[df.processingtype == args.event_type].groupby([args.mode],
                                                           as_index=False)
twc = grouped['wc'].sum().sort_values('wc', ascending=False)
bigtotal_wc = twc.wc.sum()
big_sites = twc[twc.wc > bigtotal_wc * args.wc_contrib][args.mode].tolist()

grouped = df.groupby(['processingtype', 'jeditaskid', args.mode],
                     as_index=False, sort=False)

data = defaultdict(OrderedDict)
for (p, j, s), g in grouped:
    if s not in big_sites: continue
    if p != args.event_type: continue
    if np.sum(g.cpu) < 10.: continue   # Exluding tasks with too little CPU
    x = np.average(g.cpuevt_avg, weights=g.cpu)
    y = math.sqrt(np.average((g.cpuevt_avg - x)**2, weights=g.cpu))
    data[j][s] = (x, y)

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

# Map tasks to integers
tasks = set(data.keys())
tmap = dict()
i = 0
for x in tasks:
    tmap[x] = i
    i += 1

# Initial value of the parameters
ntask = len(data)
nsite = len(smap)
a_ini = np.zeros(ntask)
for j, task in data.items():
    nt = tmap[j]
    a_ini[nt] = np.mean([b[0] for b in task.values()])
x_ini = np.hstack((a_ini, np.ones(nsite)))

print "Tasks: ", ntask
print "Sites/CPUs: ", nsite
print "Sum of initial values: ", np.sum(x_ini)

def func_val(x):
    a = x[:ntask]
    k = x[ntask:]
    total = 0.
    for j, task in data.items():   # looping on tasks
        nt = tmap[j]
        kf = np.array([k[i] for i in task.keys()])
        v = np.array([b[0] for b in task.values()])
        f_term = 1. / a[nt]**2 * (v * kf - a[nt])**2
        delta = np.sum(f_term)
        total += delta
    return total

def grad_val(x):
    a = x[:ntask]
    k = x[ntask:]
    g = np.zeros(len(x))
    for j, task in data.items():
        g_a = 0.
        nt = tmap[j]
        for i in task.keys():
            (v, _) = task[i]
            g_a += -2 * v * k[i] / a[nt]**2 * (v * k[i] / a[nt] - 1)
            g_k = 2 / a[nt]**2 * v * (k[i] * v - a[nt])
            g[ntask + i] += g_k
        g[nt] = g_a
    return g

def hess_val(x):
    a = x[:ntask]
    k = x[ntask:]
    diag = np.zeros_like(x)   # diagonal of the Hessian
    H = np.diag(diag)
    for t, task in data.items():
        h = 0.
        nt = tmap[t]
        for i in task.keys():
            (v, _) = task[i]
            h += 2 / a[nt]**4 * (3 * v**2 * k[i]**2 - 2 * v * k[i] * a[nt])
            diag[ntask + i] += 2 * v**2 / a[nt]**2
            H[nt][ntask+i] = H[ntask+i][nt] = -2 * v / a[nt]**2 * ( 2 * v * k[i] / a[nt] - 1)
        diag[nt] = h
    H = H + np.diag(diag)
    return H

Nfeval = 1

def callb(x):
    global Nfeval
    print 'Iteration #', str(Nfeval)
    Nfeval += 1

print 'Initial function value: ', func_val(x_ini)
check = optimize.check_grad(func_val, grad_val, x_ini)
print check

# Execute fit
result = optimize.minimize(func_val, x_ini, method='Newton-CG', jac=grad_val,
                           hess=hess_val, callback=callb,
                           options={'disp': True, 'maxiter': 100})

print result
x = result.x
a = x[:ntask]
k = x[ntask:]
k_err = np.zeros_like(k)

# Calculating errors
# First, calculate sigma
p = list()
for j, task in data.items():
    nt = tmap[j]
    for i in task.keys():
        (v, _) = task[i]
        p.append(v * x[ntask+i] / x[nt])
sigma = np.std(p)
print 'The value of S is: %.3f' % sigma

# Then, calculate parameter errors
for j, task in data.items():
    nt = tmap[j]
    for i in task.keys():
        (v, _) = task[i]
        k_err[i] += v**2 / a[nt]**2
k_err = sigma / np.sqrt(k_err)

# Write output to file
with open(output_file, 'wb') as csvfile:
    w = csv.writer(csvfile, delimiter=',')
    for s in sorted(smap.keys()):
        w.writerow([s, k[smap[s]], k_err[smap[s]]])
