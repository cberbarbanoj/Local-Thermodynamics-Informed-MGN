"""model.py"""

import torch
import torch.nn as nn

from src.blocks import NodeProcessorModule, EdgeProcessorModule
from src.utils.utils import decompose_graph, copy_geometric_data
from torch_geometric.data import Data
from torch_scatter import scatter_add

def instantiate_mlp(in_size, hidden_size, out_size, layers=2, lay_norm=True, dropout=0.0):

    module = [nn.Linear(in_size, hidden_size), nn.SiLU()]
    if dropout > 0.:
        module.append(nn.Dropout(dropout))
    for _ in range(layers - 2):
        module.append(nn.Linear(hidden_size, hidden_size))
        module.append(nn.SiLU())
        if dropout > 0.:
            module.append(nn.Dropout(dropout))
    module.append(nn.Linear(hidden_size, out_size))

    module = nn.Sequential(*module)

    if lay_norm:
        return nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))

    return module


class Encoder(nn.Module):

    def __init__(self,
                 edge_input_size=128,
                 node_input_size=128,
                 hidden_size=128,
                 layers=2,
                 dropout=0.0):
        super(Encoder, self).__init__()

        self.edge_encoder = instantiate_mlp(edge_input_size, hidden_size, hidden_size, layers=layers, dropout=dropout)
        self.node_encoder = instantiate_mlp(node_input_size, hidden_size, hidden_size, layers=layers, dropout=dropout)

    def forward(self, graph):

        node_attr, _, edge_attr = decompose_graph(graph)
        node_ = self.node_encoder(node_attr)
        edge_ = self.edge_encoder(edge_attr)

        return Data(x=node_, edge_index=graph.edge_index, edge_attr=edge_)


class ProcessorMessagePassing(nn.Module):

    def __init__(self,
                 hidden_size=128,
                 layers=2,
                 dropout=0.0):

        super(ProcessorMessagePassing, self).__init__()

        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size

        nb_custom_func = instantiate_mlp(nb_input_dim, hidden_size, hidden_size, layers=layers, dropout=dropout)
        eb_custom_func = instantiate_mlp(eb_input_dim, hidden_size, hidden_size, layers=layers, dropout=dropout)

        self.eb_module = EdgeProcessorModule(model=eb_custom_func)
        self.nb_module = NodeProcessorModule(model=nb_custom_func)

    def forward(self, graph):

        graph_last = copy_geometric_data(graph)

        graph = self.eb_module(graph)
        graph = self.nb_module(graph)

        edge_attr = graph_last.edge_attr + graph.edge_attr
        node_attr = graph_last.x + graph.x

        return Data(x=node_attr, edge_index=graph.edge_index, edge_attr=edge_attr)


class Decoder(nn.Module):
    def __init__(self,
                 hidden_size=128,
                 output_size=2,
                 layers=2,
                 dropout=0.0):

        super(Decoder, self).__init__()
        self.output_size = output_size
        # Get L and M output sizes
        self.size_L = int(output_size * (output_size + 1) / 2 - output_size)
        self.size_M = int(output_size * (output_size + 1) / 2)
        # Energy and entropy gradients decoders
        self.decoder_energy_grad   = instantiate_mlp(hidden_size, hidden_size, output_size, layers=layers, dropout=dropout, lay_norm=False)
        self.decoder_entropy_grad  = instantiate_mlp(hidden_size, hidden_size, output_size, layers=layers, dropout=dropout, lay_norm=False)
        # L and M operators decoders - node wise
        self.decoder_L_n = instantiate_mlp(hidden_size, hidden_size, self.size_L, layers=layers, dropout=dropout, lay_norm=False)
        self.decoder_M_n = instantiate_mlp(hidden_size, hidden_size, self.size_M, layers=layers, dropout=dropout, lay_norm=False)
        # L and M operators decoders - edge wise
        self.decoder_L_e = instantiate_mlp(3 * hidden_size, hidden_size, self.size_L, layers=layers, dropout=dropout, lay_norm=False)
        self.decoder_M_e = instantiate_mlp(3 * hidden_size, hidden_size, self.size_M, layers=layers, dropout=dropout, lay_norm=False)
        # Define additional variables for operators reparametrization
        diag = torch.eye(self.output_size, self.output_size)
        self.diag = diag[None]
        self.ones = torch.ones(self.output_size, self.output_size)

    def forward(self, graph):

        node_attr, edge_index, edge_attr = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index

        # Decode energy and entropy gradients
        dEdz = self.decoder_energy_grad(node_attr).unsqueeze(-1)
        dSdz = self.decoder_entropy_grad(node_attr).unsqueeze(-1)
        # Decode L and M operators - NODE WISE
        l_n = self.decoder_L_n(node_attr)
        m_n = self.decoder_M_n(node_attr)
        # Decode L and M operators - ELEMENT WISE
        l_e = self.decoder_L_e(torch.cat((edge_attr, node_attr[senders_idx], node_attr[receivers_idx]), dim=1))
        m_e = self.decoder_M_e(torch.cat((edge_attr, node_attr[senders_idx], node_attr[receivers_idx]), dim=1))
        # L and M operators refactorization - NODE WISE
        L_n = torch.zeros(node_attr.size(0), self.output_size, self.output_size, device=l_n.device, dtype=l_n.dtype)
        M_n = torch.zeros(node_attr.size(0), self.output_size, self.output_size, device=m_n.device, dtype=m_n.dtype)
        L_n[:, torch.tril(self.ones, -1) == 1] = l_n
        M_n[:, torch.tril(self.ones) == 1] = m_n
        L_nodes = torch.subtract(L_n, torch.transpose(L_n, 1, 2))
        M_nodes = torch.bmm(M_n, torch.transpose(M_n, 1, 2))
        # L and M operators refactorization - ELEMENT WISE
        L_e = torch.zeros(edge_attr.size(0), self.output_size, self.output_size, device=l_e.device, dtype=l_e.dtype)
        M_e = torch.zeros(edge_attr.size(0), self.output_size, self.output_size, device=m_e.device, dtype=m_e.dtype)
        L_e[:, torch.tril(self.ones, -1) == 1] = l_e
        M_e[:, torch.tril(self.ones) == 1] = m_e
        L_edges = torch.subtract(L_e, torch.transpose(L_e, 1, 2))
        M_edges = torch.bmm(M_e, torch.transpose(M_e, 1, 2))
        # Estimate NODE contribution
        L_dEdz_nodes = torch.matmul(L_nodes, dEdz)
        M_dSdz_nodes = torch.matmul(M_nodes, dSdz)
        L_dEdz_M_dSdz_nodes = L_dEdz_nodes + M_dSdz_nodes
        # Estimate EDGE contribution
        L_dEdz_edges = torch.matmul(L_edges, dEdz[senders_idx, :, :])
        M_dSdz_edges = torch.matmul(M_edges, dSdz[senders_idx, :, :])
        L_dEdz_M_dSdz_edges = L_dEdz_edges + M_dSdz_edges
        # Get output
        dzdt_net = L_dEdz_M_dSdz_nodes[:, :, 0] - scatter_add(L_dEdz_M_dSdz_edges[:, :, 0], receivers_idx, dim=0)
        # Estimate energy and entropy degeneration losses
        deg_E = torch.matmul(M_nodes, dEdz)
        deg_S = torch.matmul(L_nodes, dSdz)

        return dzdt_net, deg_E, deg_S


class EncoderProcessorDecoder(nn.Module):

    def __init__(self,
                 message_passing_num,
                 node_input_size,
                 edge_input_size,
                 output_size=3,
                 layers=2,
                 hidden_size=128,
                 shared_mlp=False,
                 dropout=0.0):

        super(EncoderProcessorDecoder, self).__init__()
        # Encoder
        self.encoder = Encoder(edge_input_size=edge_input_size, node_input_size=node_input_size, hidden_size=hidden_size, layers=layers)
        # Processor
        processor_list = [ProcessorMessagePassing(hidden_size=hidden_size, layers=layers) for _ in range(message_passing_num)]
        self.processer_list = nn.ModuleList(processor_list)
        # Decoder
        self.decoder = Decoder(hidden_size=hidden_size, output_size=output_size, layers=layers)

    def forward(self, graph, return_passes=False):
        # Encode
        graph = self.encoder(graph)
        # Process
        passes = [] if return_passes else None

        for i, model in enumerate(self.processer_list):
            graph = model(graph)
            if return_passes:
                passes.append(graph.x.clone())
        # Decode
        decoded, energy_degeneration, entropy_degeneration = self.decoder(graph)

        if return_passes:
            return decoded, energy_degeneration, entropy_degeneration, passes
        else:
            return decoded, energy_degeneration, entropy_degeneration