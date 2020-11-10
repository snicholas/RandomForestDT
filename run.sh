#!/bin/bash
echo "Model start"

if [ ! -f data/model.joblib ]; then
    echo "run training"
    python main.py 1
    echo "ended training"
fi

if [ -f data/model.joblib ]; then
    echo "run execution"
    python main.py 0
    echo "ended execution"
    cp data/model.joblib data/outputs/model.joblib
fi
echo "Model execution ended"
