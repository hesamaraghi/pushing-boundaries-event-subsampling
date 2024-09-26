from typing import Union, Iterable, List, Tuple
import os
import os.path as osp
import scipy.io as sio
import re
import random
import math
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData
import hashlib
from omegaconf import OmegaConf

try:
    from .event_filters import *
except ModuleNotFoundError:
    from event_filters import *

class FilterDataRecursive():

    def __init__(self, tau: float, filter_size: int, image_size: Tuple[int,int]):
        
        assert filter_size % 2 == 1, "Filter size must be odd"
        self.tau = tau
        self.filter_size = filter_size
        self.image_size = image_size
        self.K = filter_size // 2
        self.H, self.W = image_size
        
        sigma = filter_size / 5.0
        kernel = np.zeros((filter_size, filter_size))
        kernel[filter_size // 2, filter_size // 2] = 1
        self.gaussian_kernel = gaussian_filter(kernel, sigma)
        self.gaussian_kernel = self.gaussian_kernel / np.sum(self.gaussian_kernel)

    def __call__(self, data):
        
        self.last_time_tensor = np.full((2,self.H,self.W), float('0') , dtype=np.float32)
        self.temporal_accumulation_tensor = np.full((2,self.image_size[0],self.image_size[1]), float('0') , dtype=np.float32)

        filter_value_recursive = np.zeros(data.pos.shape[0], dtype=np.float32)

        for i ,ts in enumerate(data.pos):
    
            pp = 0 if data.x[i] < 0 else 1
    
            h = ts[-2].int()
            w = ts[-3].int()
            t = ts[-1].numpy() 
    
            h_start = max(h - self.K, 0)
            h_end = min(h + self.K, self.H-1)
            w_start = max(w - self.K, 0)
            w_end = min(w + self.K, self.W-1)

            
            # Compute the temporal lag
            temporal_lag = np.exp(- (t - self.last_time_tensor[pp,h_start:h_end+1,w_start:w_end+1])/self.tau)

            # update the last time tensor
            self.last_time_tensor[pp,h_start:h_end+1,w_start:w_end+1] = t

            # update the temporal accumulation tensor

            self.temporal_accumulation_tensor[pp,h_start:h_end+1,w_start:w_end+1] *= temporal_lag
            self.temporal_accumulation_tensor[pp,h,w] += 1

            # Compute the filter value
            filter_value_recursive[i] = np.sum(self.temporal_accumulation_tensor[pp,h_start:h_end+1,w_start:w_end+1] * self.gaussian_kernel[h_start - h + self.K:h_end + 1 - h + self.K, w_start - w + self.K:w_end +1 - w + self.K])
    
        return filter_value_recursive
    
class TemporalScaling(BaseTransform):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`
    (functional name: :obj:`normalize_scale`).
    """
    def __init__(self,cfg):
        self.temporal_scale = cfg.temporal_scale

    def __call__(self, data):

        data.pos = torch.mul(data.pos,torch.tensor([[1,1,self.temporal_scale]]))        


        return data

def filter_data(data, indices):
    num_nodes = data.num_nodes
    for key, item in data:
        if key == 'num_nodes':
            data.num_nodes = indices.size(0)
        elif bool(re.search('edge', key)):
            continue
        elif (torch.is_tensor(item) and item.size(0) == num_nodes
                and item.size(0) != 1):
            data[key] = item[indices]
    return data

def create_seed(seed_str: str) -> int:
    # Create a SHA-256 hash of the input string
    hash_object = hashlib.sha256((seed_str).encode())
    # Convert the hash to a hexadecimal string
    hex_dig = hash_object.hexdigest()
    # Convert the hexadecimal string to an integer
    seed = int(hex_dig, 16)
    return seed


class TemporalQuantization(BaseTransform):
    def __init__(self,cfg):
        self.temporal_num_bins = cfg.temporal_quantization

    def __call__(self, data):
        t = data.pos[..., -1]
        range_t = (t.max() - t.min())*(1 + 2e-3)
        min_t = t.min() - 1e-3 * range_t
        t = (t - min_t) / range_t * self.temporal_num_bins - 0.5
        data.pos[..., -1] = t.round()/self.temporal_num_bins
        return data

class RemoveOutliers(BaseTransform):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`
    (functional name: :obj:`normalize_scale`).
    """
    def __init__(self, cfg):
        self.thresh_quantile = cfg.thresh_quantile
        assert cfg.conv_vec_path is not None, "'conv_vec_path' cannot be empty!"
        self.conv_vec_path = cfg.conv_vec_path

    def __call__(self, data):

        num_nodes = data.num_nodes
        mat_path = osp.join(self.conv_vec_path,data.label[0],osp.splitext(data.file_id)[0]+'.mat')
        mat = sio.loadmat(mat_path)
        td_conv = torch.squeeze(torch.from_numpy(mat['td_conv']))
        
        if self.thresh_quantile > 1.0:
            indices = torch.argsort(td_conv, descending=True)[:self.thresh_quantile]
        else:            
            indices = torch.zeros((num_nodes,), dtype=torch.bool)
            pol_ind = torch.squeeze(data.x < 0.5)
            thr = torch.quantile(td_conv[pol_ind], self.thresh_quantile)
            indices = indices | (pol_ind & (td_conv > thr))
            pol_ind = torch.squeeze(data.x > 0.5)
            thr = torch.quantile(td_conv[pol_ind], self.thresh_quantile)
            indices = indices | (pol_ind & (td_conv > thr))

        indices = torch.nonzero(indices, as_tuple=True)[0]
        return filter_data(data, indices)
   
   
    
class FilterNodes(BaseTransform):
    r"""
    Keep the nodes that satisfy the condition defined in the filter_nodes function.
    """
    def __init__(self, cfg):

        assert cfg.filter_nodes is not None, "'filter_nodes' cannot be empty!"
        self.filter_nodes = eval(cfg.filter_nodes)

    def get_indices(self,data):
        raise NotImplementedError
    
    def __call__(self, data):

        num_nodes = data.num_nodes
        indices = self.filter_nodes(data)
        indices = torch.nonzero(indices, as_tuple=True)[0]
        return filter_data(data, indices)
    

class SpatialCentering(BaseTransform):
    r"""Centers node positions :obj:`pos` around the origin."""
    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.node_stores:
            if hasattr(store, 'pos'):
                store.pos[..., -2] = store.pos[..., -2] - store.pos[..., -2].mean()
                store.pos[..., -3] = store.pos[..., -3] - store.pos[..., -3].mean()
        return data
    
class SpatialSubsampling(BaseTransform):
    r"""Subsamples events horizontally and vertically.
    """
    def __init__(
        self,
        subsampling_ratios: tuple[int, int],
        subsampling_offsets: tuple[int, int] = None
    ):
        assert len(subsampling_ratios) == 2, 'Subsampling ratios must be a tuple of two integers.'
        assert all([isinstance(ratio, int) for ratio in subsampling_ratios]), 'Subsampling ratios must be integers.'
        assert all([ratio > 0 for ratio in subsampling_ratios]), 'Subsampling ratios must be positive integers.'
        self.subsampling_ratios = subsampling_ratios
        self.subsampling_offsets = (0,0)
        if subsampling_offsets:
            assert len(subsampling_offsets) == 2, 'Subsampling offsets must be a tuple of two integers.'
            if subsampling_offsets[0] and subsampling_offsets[1]:
                assert all([isinstance(offset, int) for offset in subsampling_offsets]), 'Subsampling offsets must be integers.'
                assert subsampling_offsets[0] >= 0, 'Subsampling offset 0 must be non negative.'
                assert subsampling_offsets[1] >= 0, 'Subsampling offset 1 must be non negative.'           
                assert subsampling_offsets[0] < subsampling_ratios[0], 'Subsampling offset 0 must be smaller than Subsampling ratio 0.'
                assert subsampling_offsets[1] < subsampling_ratios[1], 'Subsampling offset 1 must be smaller than Subsampling ratio 1.'
                self.subsampling_offsets = subsampling_offsets
            
        
    def __call__(self, data: Data) -> Data:
        pos = data.pos
        x = pos[..., -3]
        y = pos[..., -2]
        mask = torch.ones_like(x, dtype=torch.bool)
        if self.subsampling_ratios[0] > 1:
            mask = mask & (x % self.subsampling_ratios[0] == self.subsampling_offsets[0])
        if self.subsampling_ratios[1] > 1:
            mask = mask & (y % self.subsampling_ratios[1] == self.subsampling_offsets[1])
        indices = torch.nonzero(mask, as_tuple=True)[0]
        return filter_data(data, indices)
    
class TemporalSubsampling(BaseTransform):
    r"""Subsamples events temporally.
    """
    def __init__(
        self,
        subsampling_ratio: int,
        window_size: int,
        fixed_interval: bool = False,
        time_offset_coefficient: float = 0.0
        ):
        """__init__.
        Args:
            subsampling_ratio (int): Keep one out of every `subsampling_ratio` temporal intervals.
            fixed_interval (bool): If True, the interval length is fixed and equals the `window_size` parameter in milliseconds,
            if False, the interval length is calcultaed by dividing the `window_size` parameter in milliseconds by the `subsampling_ratio` parameter.
        """
        assert isinstance(subsampling_ratio, int), 'Subsampling ratio must be an integer.'
        assert subsampling_ratio > 0, 'Subsampling ratio must be a positive integer.'
        self.subsampling_ratio = subsampling_ratio
        if fixed_interval:
            self.interval_length = window_size * 1000
            self.subsampling_period = self.interval_length * self.subsampling_ratio
        else:
            self.subsampling_period = window_size * 1000
            self.interval_length = self.subsampling_period / self.subsampling_ratio
        assert time_offset_coefficient >= 0.0 and time_offset_coefficient <= 1.0, 'Time offset coefficient must be between 0.0 and 1.0.'
        self.time_offset_coefficient = time_offset_coefficient
        
    def __call__(self, data: Data) -> Data:
        pos = data.pos
        t = pos[..., -1].contiguous()
        min_time = t.min() + (self.time_offset_coefficient * (self.subsampling_period - self.interval_length))
        max_time = t.max()
        n_slices = torch.ceil((max_time - min_time) / self.subsampling_period).int()
        window_start_times = torch.arange(n_slices) * self.subsampling_period + min_time
        window_end_times = window_start_times + self.interval_length
        indices_start = torch.searchsorted(t, window_start_times)
        indices_end = torch.searchsorted(t, window_end_times)
        indices_list = [torch.arange(start, end) for start, end in zip(indices_start, indices_end)]        
        if indices_list:
            indices = torch.cat(indices_list)
        else:
            indices = torch.empty(0, dtype = torch.int64)  # Return an empty tensor
        return filter_data(data, indices)
    
class TemporalSubsamplingRandomOffset(TemporalSubsampling):
    r"""Subsamples events horizontally and vertically, with random offset.
    """
    def __init__(
        self,
        cfg_transform,
        subsampling_ratio: int,
        window_size: int,
        fixed_interval: bool = False,
        ):
        super().__init__(
                    subsampling_ratio = subsampling_ratio,
                    window_size = window_size,
                    fixed_interval = fixed_interval
                )
        # fixed subsampling initialization
        self.fixed_subsampling = False
        if "fixed_sampling" in cfg_transform and cfg_transform["fixed_sampling"]["transform"] is True:
            self.fixed_subsampling = True
            if "seed_str" in cfg_transform["fixed_sampling"] and cfg_transform["fixed_sampling"]["seed_str"] is not None:   
                self.seed_str = str(cfg_transform["fixed_sampling"]["seed_str"])
            else:
                self.seed_str = 'fixed_subsampling' 
    
    def __call__(self, data: Data) -> Data:
        if self.fixed_subsampling:
            seed = create_seed(self.seed_str + '_' + data.label[0] + '_' + data.file_id)
            torch_rng = torch.Generator().manual_seed(seed % (2**32))
            self.time_offset_coefficient = torch.rand(1, generator=torch_rng)[0]
        else:
            self.time_offset_coefficient = torch.rand(1)[0]
        return super().__call__(data)

class FixedSubsampling(BaseTransform):
    r"""Fixed num subsampling of nodes.
    """
    def __init__(
        self,
        cfg,
        replace: bool = True,
        allow_duplicates: bool = False,
    ):
        self.num = cfg.num_events_per_sample
        if "seed_str" in cfg["fixed_sampling"] and cfg.fixed_sampling.seed_str is not None:   
            self.seed_str = str(cfg.fixed_sampling.seed_str)
        else:
            self.seed_str = 'fixed_subsampling' 
        self.replace = replace
        self.allow_duplicates = allow_duplicates

    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        seed = create_seed(self.seed_str + '_' + data.label[0] + '_' + data.file_id)
        rng = np.random.default_rng(seed)
        torch_rng = torch.Generator().manual_seed(seed % (2**32))

        if self.replace:
            choice = rng.choice(num_nodes, self.num, replace=True)
            choice = torch.from_numpy(choice).to(torch.long)
        elif not self.allow_duplicates:
            choice = torch.randperm(num_nodes, generator=torch_rng)[:self.num]
        else:
            choice = torch.cat([
                torch.randperm(num_nodes, generator=torch_rng)
                for _ in range(math.ceil(self.num / num_nodes))
            ], dim=0)[:self.num]

        return filter_data(data, choice)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num}, replace={self.replace})'

    
    

class SpatialScaling(BaseTransform):
    r"""Scales node positions by a randomly sampled factor :math:`s` within a
    given interval, *e.g.*, resulting in the transformation matrix

    .. math::
        \begin{bmatrix}
            s & 0 & 0 \\
            0 & s & 0 \\
            0 & 0 & 0 \\
        \end{bmatrix}

    for three-dimensional positions.

    Args:
        scales (tuple): scaling factor interval, e.g. :obj:`(a, b)`, then scale
            is randomly sampled from the range
            :math:`a \leq \mathrm{scale} \leq b`.
    """
    def __init__(self, cfg):
        assert len(cfg.scale_limits) == 2
        self.scales = cfg.scale_limits

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        data.pos[...,-2] = data.pos[...,-2] * scale
        data.pos[...,-3] = data.pos[...,-3] * scale
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.scales})'


class AddEdgeAttr(BaseTransform):

    def __init__(self, cfg):
        # self.norm = norm
        # self.max = max_value
        self.cat = cfg.cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos[:, :2], data.edge_attr
        data.pos = pos
        cart = torch.abs(pos[row] - pos[col])
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        # if self.norm and cart.numel() > 0:
        #     max_value = cart.abs().max() if self.max is None else self.max
        #     cart = cart / (2 * max_value) + 0.5

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')
        
        
        
class ShiftAndFlip(BaseTransform):

    def __init__(self, cfg):
        self.max_shift = cfg.max_shift
        self.p = cfg.p
        if cfg.resolution is None:
            self.resolution = (180,240)
        else:
            self.resolution = cfg.resolution

    def __call__(self, data):
        H, W = self.resolution
        x_shift, y_shift = torch.randint(-self.max_shift, self.max_shift + 1, (2,))
        data.pos[..., -3] += x_shift
        data.pos[..., -2] += y_shift
        
        indices = (data.pos[..., -3] >= 0) & (data.pos[..., -3] < W) & (data.pos[..., -2] >= 0) & (data.pos[..., -2] < H)
        indices = torch.nonzero(indices, as_tuple=True)[0]
        data =  filter_data(data, indices)
        if torch.rand(1) < self.p:
            data.pos[..., -3] = W - 1 - data.pos[..., -3]
            
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(max_shift={self.max_shift}, '
                f'p={self.p})')
        
        
        
class VaryingSamplingPoints(BaseTransform):
    r"""Samples a random number of points and features from the nodes.
    Args:
        range_num (tuple): minimun and maximum for random number of points to sample.
        replace (bool, optional): If set to :obj:`False`, samples points
            without replacement. (default: :obj:`True`)
        allow_duplicates (bool, optional): In case :obj:`replace` is
            :obj`False` and :obj:`num` is greater than the number of points,
            this option determines whether to add duplicated nodes to the
            output points or not.
            In case :obj:`allow_duplicates` is :obj:`False`, the number of
            output points might be smaller than :obj:`num`.
            In case :obj:`allow_duplicates` is :obj:`True`, the number of
            duplicated points are kept to a minimum. (default: :obj:`False`)
    """
    def __init__(
        self,
        range_num: tuple,
        replace: bool = True,
        allow_duplicates: bool = False,
        inverse_sampling: bool = False,
    ):
        assert len(range_num) == 2 and range_num[0] < range_num[1]
        self.num_list = np.arange(*range_num)
        if inverse_sampling:
            weights = 1 / self.num_list
            self.weights = weights / weights.sum()
        else:
            self.weights = None
        self.replace = replace
        self.allow_duplicates = allow_duplicates

    def __call__(self, data: Data) -> Data:
        
        self.num = np.random.choice(self.num_list, size=1, p=self.weights)[0]
        num_nodes = data.num_nodes

        if self.replace:
            choice = np.random.choice(num_nodes, self.num, replace=True)
            choice = torch.from_numpy(choice).to(torch.long)
        elif not self.allow_duplicates:
            choice = torch.randperm(num_nodes)[:self.num]
        else:
            choice = torch.cat([
                torch.randperm(num_nodes)
                for _ in range(math.ceil(self.num / num_nodes))
            ], dim=0)[:self.num]

        return filter_data(data, choice)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(({self.range_num[0]},{self.range_num[1]}), replace={self.replace})'
    
    
    
    

class DropEveryNthEvent(BaseTransform):
    """ From tonic.transforms
    
    Deterministically drops every nth event for every spatial location x (and potentially y).

    Parameters:
        n (int): The event stream for each x/y location is reduced to 1/n.

    Example:
        >>> transform = tonic.transforms.Decimation(n=5)
    """

    def __init__(
        self,
        n: int,
    ):
        assert n > 0, "n has to be an integer greater than zero."
        self.n = n
        
    def __call__(self, data: Data) -> Data:
        
        max_x = data.pos[..., -3].max().int().item()
        max_y = data.pos[..., -2].max().int().item()

        indices = []
        memory = torch.zeros((max_x + 1, max_y + 1))

        # Sort the data.pos[..., -1] array in ascending order
        sorted_indices = torch.argsort(data.pos[..., -1])
        data.pos = data.pos[sorted_indices]
        data.x = data.x[sorted_indices]

        for event_num in range(data.num_nodes):
            event_x, event_y = data.pos[event_num,-3].int(), data.pos[event_num,-2].int()
            memory[event_x, event_y] += data.x[event_num,0]
            if torch.abs(memory[event_x, event_y]) >= self.n:
                memory[event_x, event_y] = 0
                indices.append(event_num)
        
        return filter_data(data, indices)


class DropEventRandomly(BaseTransform):

    def __init__(self, cfg):
        cfg_dict = OmegaConf.to_object(cfg)
        self.fixed_subsampling = False
        if "fixed_sampling" in cfg_dict and cfg.fixed_sampling.transform is True:
            self.fixed_subsampling = True
            if "seed_str" in cfg["fixed_sampling"] and cfg.fixed_sampling.seed_str is not None:   
                self.seed_str = str(cfg.fixed_sampling.seed_str)
            else:
                self.seed_str = 'fixed_subsampling' 
        self.p = cfg.random_ratio_subsampling
        
    def __call__(self, data: Data) -> Data:   
        n_events = data.num_nodes
        if self.fixed_subsampling:
            seed = create_seed(self.seed_str + '_' + data.label[0] + '_' + data.file_id)
            torch_rng = torch.Generator().manual_seed(seed % (2**32))
            indices = torch.where(torch.rand(n_events, generator=torch_rng) < self.p)[0]
        else:
            indices = torch.where(torch.rand(n_events) < self.p)[0]
            
        return filter_data(data, indices)   
    
    
class SpatioTemporalFilteringSubsampling(BaseTransform, FilterDataRecursive):
    r"""Subsampling the event video using spatio-temporal filter values."""
    
    def __init__(self, cfg_all, cfg_transform):
        
        if not isinstance(cfg_all, dict):
            cfg_all = OmegaConf.to_object(cfg_all)
        if not isinstance(cfg_transform, dict):
            cfg_transform = OmegaConf.to_object(cfg_transform)
        
        assert "tau" in cfg_transform['spatiotemporal_filtering_subsampling'], "tau must be provided in the transform config"
        assert "filter_size" in cfg_transform['spatiotemporal_filtering_subsampling'], "filter_size must be provided in the transform config"
        assert "sampling_threshold" in cfg_transform['spatiotemporal_filtering_subsampling'], "sampling_threshold must be provided in the transform config"
        assert "normalization_length" in cfg_transform['spatiotemporal_filtering_subsampling'], "normalization_length must be provided in the transform config"
        
        tau = cfg_transform['spatiotemporal_filtering_subsampling']['tau']
        filter_size = cfg_transform['spatiotemporal_filtering_subsampling']['filter_size']
        sampling_threshold = cfg_transform['spatiotemporal_filtering_subsampling']['sampling_threshold']
        normalization_length = cfg_transform['spatiotemporal_filtering_subsampling']['normalization_length']
        
        assert isinstance(tau, int), "tau must be an integer"
        assert isinstance(filter_size, int), "filter_size must be an integer"
        tau = int(tau * 1000)
        image_size = cfg_all["dataset"]["image_resolution"]
        assert len(image_size) == 2, "image_resolution must be a tuple of two integers"
        assert cfg_all["dataset"]["name"] is not None, "dataset name must be provided"
        assert cfg_all["dataset"]["dataset_path"] is not None, "dataset path must be provided"
        self.batch_list_dir = osp.join(cfg_all["dataset"]["dataset_path"], "filter_values", f"tau_{tau}_filter_size_{filter_size}")
        
        
        FilterDataRecursive.__init__(self, tau, filter_size, image_size)
        
        # filtering parameters
        self.normalization_length = normalization_length
        
        # subsampling parameters
        if normalization_length:
            assert isinstance(normalization_length, int), "normalization_length must be an integer"
        self.sampling_threshold = sampling_threshold

        # fixed subsampling initialization
        self.fixed_subsampling = False
        if "fixed_sampling" in cfg_transform and cfg_transform["fixed_sampling"]["transform"] is True:
            self.fixed_subsampling = True
            if "seed_str" in cfg_transform["fixed_sampling"] and cfg_transform["fixed_sampling"]["seed_str"] is not None:   
                self.seed_str = str(cfg_transform["fixed_sampling"]["seed_str"])
            else:
                self.seed_str = 'fixed_subsampling' 
        
    def __call__(self, data: Data) -> Data:   
        
        filter_values = self.get_filter_values(data)
        assert len(filter_values) == data.num_nodes, "Filter values must have the same length as the number of nodes in the data"
        if self.normalization_length:
            filter_values = self.filter_values_normalizing(filter_values)
        
        # subsampling  
        n_events = data.num_nodes
        if self.fixed_subsampling:
            seed = create_seed(self.seed_str + '_' + data.label[0] + '_' + data.file_id)
            torch_rng = torch.Generator().manual_seed(seed % (2**32))
            indices = torch.where(torch.rand(n_events, generator=torch_rng) < self.sampling_threshold * filter_values)[0]
        else:
            indices = torch.where(torch.rand(n_events) < self.sampling_threshold * filter_values)[0]
            
        return filter_data(data, indices)   
  
    def get_filter_values(self, data):
        
        class_name = data.label[0]     
        filter_value_file = osp.join(self.batch_list_dir, class_name)
        filter_value_file = osp.join(filter_value_file, "filter_values_" + osp.splitext(data.file_id)[0] + ".pt")
        
        if not osp.exists(filter_value_file):
            print(f"Filter values for {data.file_id} do not exist!")
            print(f"Creating filter values for {data.file_id}!")
            if not osp.exists(osp.dirname(filter_value_file)):
                os.makedirs(osp.dirname(filter_value_file))
            
            sorted_indices = torch.argsort(data.pos[..., -1])
            data.pos = data.pos[sorted_indices]
            data.x = data.x[sorted_indices]

            filter_values = FilterDataRecursive.__call__(self, data)

            # To find reverse_indices such that A = B[reverse_indices]
            reverse_indices = torch.zeros_like(sorted_indices)
            reverse_indices[sorted_indices] = torch.arange(len(sorted_indices))

            filter_values = filter_values[reverse_indices]
            filter_values = torch.tensor(filter_values)
            
            print(f"Saving filter values for {data.file_id} to {filter_value_file}")
            torch.save(filter_values, filter_value_file)
        
        else:
            filter_values = torch.load(filter_value_file)
            # print(f"Loaded filter values from {filter_value_file}")
        
        return filter_values
    
    def filter_values_normalizing(self, filter_values):
        
        filter_values_normalized = torch.zeros_like(filter_values)
        for i, ev in enumerate(filter_values):
            start = max(0, i - self.normalization_length)
            chunk_part = filter_values[start:i+1]
            chunk_part_normalized = (chunk_part - torch.min(chunk_part)) / (torch.max(chunk_part) - torch.min(chunk_part) + 1e-6)
            filter_values_normalized[i] = chunk_part_normalized[-1]
        return filter_values_normalized
    