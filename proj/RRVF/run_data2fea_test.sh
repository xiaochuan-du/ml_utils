#/bin/bash
PYTHONPATH='.' luigi --module data2fea_exp Period --local-scheduler --workers=4 --end-date 2017-04-23 --ndays 39
