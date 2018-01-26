#!/bin/bash
docker run -it -v $PWD:/src -w /src dy:dev python3 predict_prop_mul.py
