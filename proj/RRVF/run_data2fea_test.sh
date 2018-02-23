#/bin/bash
PYTHONPATH='.' luigi --module data2fea_exp Period --local-scheduler --workers=4 --end-date 2017-04-22 --ndays 39
