"""normalization.py"""

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    def __init__(self, size, max_accumulations=10**6, std_epsilon=1e-8, name='Normalizer', device='cuda'):
        super(Normalizer, self).__init__()
        self.name=name
        self.max_accumulations = max_accumulations
        self._frozen = False

        self.register_buffer("std_epsilon", torch.tensor(std_epsilon, dtype=torch.float32, device=device))
        self.register_buffer("acc_count", torch.tensor(0.0, dtype=torch.float32, device=device))
        self.register_buffer("num_accumulations", torch.tensor(0.0, dtype=torch.float32, device=device))
        self.register_buffer("acc_sum", torch.zeros((1, size), dtype=torch.float32, device=device))
        self.register_buffer("acc_sum_squared", torch.zeros((1, size), dtype=torch.float32, device=device))
    
    def forward(self, batched_data, accumulate=True):
        if hasattr(self, '_frozen') and self._frozen:
            accumulate = False
        if accumulate and self.num_accumulations < self.max_accumulations:
            self._accumulate(batched_data.detach())
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data):
        """Accumulate statistics from batched data."""
        count = batched_data.shape[0]
        data_sum = torch.sum(batched_data, dim=0, keepdim=True)
        squared_data_sum = torch.sum(batched_data**2, dim=0, keepdim=True)

        with torch.no_grad():
            self.acc_sum += data_sum
            self.acc_sum_squared += squared_data_sum
            self.acc_count += count
            self.num_accumulations += 1
    
    def freeze(self):
        self._frozen = True

    def _mean(self):
        safe_count = torch.maximum(self.acc_count, torch.tensor(1.0, dtype=self.acc_count.dtype, device=self.acc_count.device))
        return self.acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(self.acc_count, torch.tensor(1.0, dtype=self.acc_count.dtype, device=self.acc_count.device))
        std = torch.sqrt(self.acc_sum_squared / safe_count - self._mean() ** 2)
        return torch.maximum(std, self.std_epsilon)
    
    def get_variable(self):
        return {
            '_max_accumulations': self.max_accumulations,
            '_std_epsilon': self.std_epsilon.clone(),
            '_acc_count': self.acc_count.clone(),
            '_num_accumulations': self.num_accumulations.clone(),
            '_acc_sum': self.acc_sum.clone(),
            '_acc_sum_squared': self.acc_sum_squared.clone(),
            'name': self.name
        }

    def load_variable(self, state_dict):
        with torch.no_grad():
            self.std_epsilon.copy_(state_dict['_std_epsilon'])
            self.acc_count.copy_(state_dict['_acc_count'])
            self.num_accumulations.copy_(state_dict['_num_accumulations'])
            self.acc_sum.copy_(state_dict['_acc_sum'])
            self.acc_sum_squared.copy_(state_dict['_acc_sum_squared'])