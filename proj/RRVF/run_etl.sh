#/bin/bash
PYTHONPATH='.' luigi --module data2fea AggTsFeas --local-scheduler --workers=1