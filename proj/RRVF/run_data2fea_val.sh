#/bin/bash
PYTHONPATH='.' luigi --module data2fea_exp Period --local-scheduler --workers=3 --end-date 2017-03-12 --ndays 42
