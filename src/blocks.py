"""blocks.py"""

import torch
import torch.nn as nn
from torch_scatter import scatter_add
from src.utils.utils import decompose_graph
from torch_geometric.data import Data


class EdgeProcessorModule(nn.Module):
    def __init__(self, model=None):

        super(EdgeProcessorModule, self).__init__()
        self.net = model

    def forward(self, graph):

        node_attr, edge_index, edge_attr = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index
        edges_to_collect = []

        senders_attr   = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_attr)


        collected_edges = torch.cat(edges_to_collect, dim=1)

        edge_attr_ = self.net(collected_edges)

        return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr_)


class NodeProcessorModule(nn.Module):
    def __init__(self, model=None):

        super(NodeProcessorModule, self).__init__()
        self.net = model

    def forward(self, graph):

        node_attr, edge_index, edge_attr = decompose_graph(graph)
        nodes_to_collect = []

        _, receivers_idx   = edge_index
        num_nodes          = graph.num_nodes
        agg_received_edges = scatter_add(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)

        nodes_to_collect.append(node_attr)
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)

        node_attr = self.net(collected_nodes)

        return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)

