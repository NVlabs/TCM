# ---------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This file has been taken from  EDM.
#
# Source:
# https://github.com/NVlabs/edm/blob/main/torch_utils (EDM)
#
# The license for these can be found in license/ directory.
# ---------------------------------------------------------------

import re
import contextlib
import numpy as np
import torch
import warnings
import dnnlib
import matplotlib.pyplot as plt
import scipy.stats as stats

#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

#----------------------------------------------------------------------------
# Replace NaN/Inf with specified numerical values.

try:
    nan_to_num = torch.nan_to_num # 1.8.0a0
except AttributeError:
    def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None): # pylint: disable=redefined-builtin
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)


# Variant of constant() that inherits dtype and device from the given
# reference tensor by default.

def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(value, shape=shape, dtype=dtype, device=device, memory_format=memory_format)
#----------------------------------------------------------------------------
# Symbolic assert.

try:
    symbolic_assert = torch._assert # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert # 1.7.0

#----------------------------------------------------------------------------
# Context manager to temporarily suppress known warnings in torch.jit.trace().
# Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672

@contextlib.contextmanager
def suppress_tracer_warnings():
    flt = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)

#----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().

def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}')
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(size, torch.as_tensor(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')

#----------------------------------------------------------------------------
# Function decorator that calls torch.autograd.profiler.record_function().

def profiled_function(fn):
    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)
    decorator.__name__ = fn.__name__
    return decorator

#----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

#----------------------------------------------------------------------------
# Utilities for operating with torch.nn.Module parameters and buffers.

def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

@torch.no_grad()
def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all), f"Missing source tensor: {name}"
        if name in src_tensors:
            tensor.copy_(src_tensors[name])

#----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.

@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield

#----------------------------------------------------------------------------
# Check DistributedDataParallel consistency across processes.

def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert (tensor == other).all(), fullname

#----------------------------------------------------------------------------
# Print summary table of module hierarchy.

def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]
    def pre_hook(_mod, _inputs):
        nesting[0] += 1
    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(dnnlib.EasyDict(mod=mod, outputs=outputs))
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs

#----------------------------------------------------------------------------

def get_t(i, sigma_max=80, sigma_min=0.002, rho=7):
    """
    Calculate edm-style t given i in the range [0, 1]. get_t(1) = sigma_max, and get_t(0) = sigma_min
    """
    t = (sigma_max ** (1 / rho) + (1 - i) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return t
    
class LoggingT:
    def __init__(self, n_bins=100, alpha=1.):
        self.n_bins = n_bins
        self.alpha = alpha  # Moving average rate
        self.bin_edges = [get_t(i / n_bins) for i in range(n_bins + 1)]  # Store bin edges
        self.bin_edges[-1] = 100
        self.bin_edges[0] = 0.
        self.bin_edges = np.array(self.bin_edges)
        self.log_bin = np.zeros(n_bins)  # Initialize moving average for the quantity
        self.iteration = 0

    def get_bin_idx(self, t_values, values):
        # If torch tensors are used, convert them to numpy arrays
        if isinstance(t_values, torch.Tensor):
            t_values = t_values.detach().cpu().numpy()
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()

        # Check if any t_values are outside the bounds and print them if they are
        if not (np.all(t_values > self.bin_edges[0]) and np.all(t_values < self.bin_edges[-1])):
            # Identify values outside the bounds
            out_of_bounds_values = t_values[(t_values < self.bin_edges[0]) | (t_values > self.bin_edges[-1])]
            raise ValueError(f"t_values contain elements outside the defined range of bin edges: {out_of_bounds_values}")
        bin_idx = np.digitize(t_values, self.bin_edges[1:])  # Determine bin indices
        return bin_idx
            
    def update(self, t_values, values):
        # If torch tensors are used, convert them to numpy arrays
        if isinstance(t_values, torch.Tensor):
            t_values = t_values.detach().cpu().numpy()
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()

        bin_idx = self.get_bin_idx(t_values, values)
        # Apply updates using np.where

        sums = np.bincount(bin_idx, weights=values, minlength=self.n_bins)
        counts = np.bincount(bin_idx, minlength=self.n_bins)

        avg_values = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)

        # Apply updates to the indices where counts are non-zero
        update_indices = np.where(counts != 0)
        self.log_bin[update_indices] = (1 - self.alpha) * self.log_bin[update_indices] + self.alpha * avg_values[update_indices]
        self.iteration += 1
    
    def plot(self, save_path):
        """
        Plot the logged quantity as a histogram.
        Parameters:
        - save_path: Path to save the plot.
        """
        plt.figure(figsize=(12, 6))  # Increase figure size for better visibility
        
        # Create a histogram plot
        plt.hist(self.bin_edges[:-1], bins=self.bin_edges, weights=self.log_bin, edgecolor='black', alpha=0.7)
        
        plt.xlabel('t')
        # Set x-ticks to bin edges and format the labels
        plt.xscale('log')

        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))

        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
    def warm_up(self):
        return self.iteration < 100

        
    
    def get_log(self):
        return self.log_bin


def get_edm_cout(sigma, sigma_data=0.5):
    return sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()

def truncated_normal(num_samples, mu, sigma, lower, upper):
    # Sample from N(mu, sigma^2) truncated to the range [lower, upper]
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    x = stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=num_samples)
    return x

def truncated_t(num_samples, mu, sigma, lower, upper, df=2):
    # Sample from truncated student_t
    lower_adjusted = (lower - mu) / sigma
    upper_adjusted = (upper - mu) / sigma
    
    # Get the CDF values for the adjusted bounds
    a = stats.t.cdf(lower_adjusted, df)
    b = stats.t.cdf(upper_adjusted, df)
    
    # Sample from the uniform distribution within the adjusted CDF bounds
    u = np.random.uniform(a, b, num_samples)
    
    # Get the PPF and then adjust it to match the desired mean and scale
    x = stats.t.ppf(u, df) * sigma + mu
    return x
