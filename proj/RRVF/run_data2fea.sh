#/bin/bash
PYTHONPATH='.' luigi --module data2fea_exp AggTsFeas --local-scheduler --workers=4 --pivot-date '2017-03-22' --end-date 2017-04-22 --start-date 2016-01-01 --date-step 7 --days-in-label 39 --min-num-in-stat-set 37
