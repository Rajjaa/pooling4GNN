#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
# python main.py --cfg configs/pyg/example_node.yaml --repeat 3 # node classification
# python main.py --cfg configs/pyg/example_link.yaml --repeat 3 # link prediction
# python main.py --cfg configs/pyg/example_graph.yaml --repeat 1 # graph classification
# python main.py --cfg configs/pyg/global_attention.yaml --repeat 2 # graph classification with global pooling
# python main.py --cfg configs/pyg/sort_pool.yaml --repeat 1 # graph classification with global pooling
python main.py --cfg configs/pyg/mincutpool.yaml --repeat 1 # graph classification with global pooling