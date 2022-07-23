export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


#CUDA_VISIBLE_DEVICES=0 tell evaluate expt/goodnews/5_transformer_roberta/config.yaml -m expt/goodnews/5_transformer_roberta/serialization/best.th
python scripts/compute_metrics.py -c /home/jupyter/data/GoodNews/goodnews/name_counters.pkl expt/goodnews/5_transformer_roberta/serialization/generations.jsonl
