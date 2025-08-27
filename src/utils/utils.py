"""utils.py"""

import os
import shutil
import argparse
import enum
from torch_geometric.data import Data
import re

def str2bool(v):
    # Code from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_error(error):
    print('State Variable (L2 relative error)')
    lines = []

    for key in error.keys():
        e = error[key]
        # error_mean = sum(e) / len(e)
        line = '  ' + key + ' = {:1.2e}'.format(e)
        print(line)
        lines.append(line)
    return lines


def create_folder(output_dir_exp):

    os.makedirs(output_dir_exp, exist_ok=True)

    return output_dir_exp

def generate_folder(output_dir_exp, pahtDInfo, pathWeights):
    if os.path.exists(output_dir_exp):
        print("The experiment path exists.")
        action = input("¿Would you like to create a new one (c) or overwrite (o)?")
        if action == 'c':
            output_dir_exp = output_dir_exp + '_new'
            os.makedirs(output_dir_exp, exist_ok=True)
    else:
        os.makedirs(output_dir_exp, exist_ok=True)

    shutil.copyfile(os.path.join('src', 'gnn_global.py'), os.path.join(output_dir_exp, 'gnn_global.py'))
    shutil.copyfile(os.path.join('data', 'jsonFiles', pahtDInfo),
                    os.path.join(output_dir_exp, os.path.basename(pahtDInfo)))
    shutil.copyfile(os.path.join('data', 'weights', pathWeights),
                    os.path.join(output_dir_exp, os.path.basename(pathWeights)))
    return output_dir_exp


def compare_metrics():
    folder_path = os.path.join('../../outputs', 'runs')
    for i in os.listdir(folder_path):
        os.path.join(folder_path, i, 'metrics.txt')

def parse_metrics(file_path):
    keys = []
    metrics = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'(\S+)\s*=\s*([0-9eE\+\-\.]+)\n', line[2:])
            if match:
                key, value = match.groups()
                # metrics[key] = float(value)
                metrics.append(float(value))
                keys.append(key)
            else:
                key, value = line[2:].split('=')
                # metrics[key] = float(value)
                metrics.append(float(value))
                keys.append(key)
    return metrics, keys

def print_table(headers, data):
    row_format = "{:<20}" + "{:<15}" * len(headers[1:])
    print(row_format.format(*headers))
    try:
        for row in data:
            print(row_format.format(row[0], *row[1].values()))
    except:
        print()


class NodeType(enum.IntEnum):
  NORMAL = 0
  OBSTACLE = 1
  AIRFOIL = 2
  HANDLE = 3
  INFLOW = 4
  OUTFLOW = 5
  WALL_BOUNDARY = 6
  SIZE = 9

    
def decompose_graph(graph):

    return (graph.x, graph.edge_index, graph.edge_attr)

def copy_geometric_data(graph):
    node_attr, edge_index, edge_attr = decompose_graph(graph)

    ret = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)

    return ret


class lambdaLinearScheduler:
    def __init__(self, start_value, end_value, start_epoch, warmup):
        self.start_value = start_value
        self.end_value = end_value
        self.start_epoch = start_epoch
        self.warmup = warmup

    def getLambda(self, epoch):
        if epoch < self.start_epoch:
            return 0
        else:
            if epoch < self.warmup:
                lambda_deg = self.start_value + (self.end_value - self.start_value) * (epoch - self.start_epoch) / (self.warmup - self.start_epoch)
                return lambda_deg
            else:
                return self.end_value
