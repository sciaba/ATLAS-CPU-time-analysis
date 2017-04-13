#! /usr/bin/env python

import csv
import sys
import json
from collections import defaultdict
import pandas as pd
from pandas import Series, DataFrame

# Read ATLAS topology
cs2goc = dict()
with open('topology.json') as f:
    topo = json.load(f)

for s in topo.values():
    cs2goc[s['panda_resource']] = s['gocname']

input = 'capacities.csv'
data = defaultdict(list)
index = list()
with open(input) as cvsfile:
    r = csv.reader(cvsfile, delimiter=',')
    for line in r:
        site = line[0]
        tot_cpu = float(line[2])
        tot_hs06 = float(line[3])
        index.append(site)
        data['tot_cpu'].append(tot_cpu)
        data['tot_hs06'].append(tot_hs06)

df_rebus = pd.DataFrame(data, index=index)
df_rebus['power'] = df_rebus['tot_hs06'] / df_rebus['tot_cpu']

print df_rebus

input = 'results.csv'
data = defaultdict(list)
index = list()
with open(input) as cvsfile:
    r = csv.reader(cvsfile, delimiter=',')
    for line in r:
        cs = line[0]
        k = float(line[1])
        if cs in cs2goc.keys():
            site = cs2goc[cs]
            power = -1.
            if site in df_rebus.index:
                power = df_rebus.loc[site].power
                index.append(cs)
                data['kfact'].append(k)
                data['power'].append(power)

df_kfact = pd.DataFrame(data, index=index)
df_kfact['ratio'] = df_kfact['kfact'] / df_kfact['power'] * 10.

print df_kfact
