# ATLAS-CPU-time-analysis
Jupyter notebook for ATLAS job analysis.

## How to retrieve the ATLAS topology
```
wget http://atlas-agis-api.cern.ch/request/pandaqueue/query/list/?json&preset=schedconf.all&vo_name=atlas
```
and save to `topology.json`.

## How to retrieve the REBUS capacity
Go to `https://wlcg-rebus.cern.ch/apps/capacities/sites/`, select the desired VO, year and month and get the table in CSV format. Save to a file called `capacities.csv`.
