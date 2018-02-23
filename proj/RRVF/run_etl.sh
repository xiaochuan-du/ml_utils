#/bin/bash
PYTHONPATH='.' luigi --module data2fea DataSplits --local-scheduler --workers=1
