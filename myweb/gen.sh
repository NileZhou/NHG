#!/bin/bash

cd ../
python pipeline.py -article_path $1 -model_prefix $2
