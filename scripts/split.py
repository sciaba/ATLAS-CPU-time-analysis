#! /bin/env python

import sys, re, random

input = sys.argv[1]
no_outputs = 10
g = list()
output_pattern = 'task_cpu_sub_%s.csv'
p = re.compile('\d+')
jtask_old = ''
for i in range(no_outputs):
    g.append(open(output_pattern % (str(i)), 'w'))
    
with open(input) as f:
    data = f.readlines()
    data.sort()
    for line in data:
        jtask = p.match(line).group(0)
        if jtask != jtask_old:
            jtask_old = jtask
            i = random.randint(0, no_outputs-1)
        g[i].write(line)

for i in range(no_outputs):
    g[i].close()
