# ---------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from ECT codebase, which built upon EDM.
#
# Source:
# https://github.com/NVlabs/edm/blob/main/training/generate.py (EDM)
# https://github.com/locuslab/ect/blob/main/training/ct_eval.py (ECT)
#
# The license for these can be found in license/ directory.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------


import os
import re
import json
import click

import pickle
import psutil
import functools
import PIL.Image
import numpy as np

import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

from metrics import metric_main
from torchvision.utils import save_image

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture',                                       type=click.Choice(['ddpmpp', 'ncsnpp', 'adm', 'edm2-cifar-s', 'edm2-cifar-m', 'edm2-img64-s', 'edm2-img64-m', 'edm2-img64-l', 'edm2-img64-xl' ]), default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm',       type=click.Choice(['vp', 've', 'edm', 'ct']), default='ct', show_default=True)

# Hyperparameters.
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0., show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Model Hyperparameters
@click.option('--mean',          help='P_mean of Log Normal Distribution', metavar='FLOAT',         type=click.FloatRange(), default=-1.1, show_default=True)
@click.option('--std',           help='P_std of Log Normal Distribution', metavar='FLOAT',          type=click.FloatRange(), default=2.0, show_default=True)
@click.option('--scale',         help='Fourier Scale for NCSN++', metavar='FLOAT',                  type=click.FloatRange(min=0), default=1., show_default=True)
@click.option('--learnable_scale', help='Learnable Scale for NCSN++', metavar='BOOL',               type=bool, default=False, show_default=True)
@click.option('--attn_type',     help='Attention type', metavar='STR',                              type=click.Choice(['dot', 'l2', 'none']), default='dot', show_default=True)   
@click.option('--emb_norm',    help='Embedding normalization', metavar='BOOL',                    type=bool, default=False, show_default=True)


@click.option('--scheduler',     help='Type of consistency scheduler', metavar='STR',               type=click.Choice(['logsnr', 'power', 'sigmoid']), default='sigmoid', show_default=True)
@click.option('--double',        help='How often to save latest checkpoints', metavar='TICKS',      type=click.IntRange(min=1), default=500, show_default=True)

@click.option('-q',              help='Decay Factor', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1.4, show_default=True)
@click.option('-c',              help='Constant c for Huber Loss', metavar='FLOAT',                 type=click.FloatRange(), default=0.0, show_default=True)
@click.option('-k',              help='Consistency condition hyperparams.', metavar='FLOAT',        type=click.FloatRange(), default=8.0, show_default=True)
@click.option('-b',              help='Consistency condition hyperparams.', metavar='FLOAT',        type=click.FloatRange(), default=1.0, show_default=True)
@click.option('--cut',           help='Cutoff value.', metavar='FLOAT',                             type=click.FloatRange(), default=4.0, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--resume',        help='Load network pickle', metavar='PKL|URL',   type=str)
@click.option('-n', '--dry_run', help='Print training options and exit',                            is_flag=True)

# Evaluation
@click.option('--mid_t',         help='Sampler steps [default working value: 0.821]',                             multiple=True, default=None)
@click.option('--metrics',       help='Comma-separated list or "none" [default: fid50k_full]',      type=CommaSeparatedList(), default='fid50k_full')
@click.option('--im_dir',        help='Path to the synthetic image directory, for distillation error evaluation', metavar='DIR', type=str, default=None)
@click.option('--latent_dir',        help='Path to the noise directory, for distillation error evaluation', metavar='DIR', type=str, default=None)
@click.option('--save_pth',        help='Save the .pth file',                                       is_flag=True)
@click.option('--num_samples',   help='Number of samples to generate', metavar='INT',               type=click.IntRange(min=1), default=50000, show_default=True)
@click.option('--class_label', help='Specific class label to generate (0-999)', type=click.IntRange(0, 999), default=None)


def main(**kwargs):
    """Train ECMs using the techniques described in the 
    blog "Consistency Models Made Easy".
    """   
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.network_kwargs = dnnlib.EasyDict()

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2], attn_type=opts.attn_type, emb_norm=opts.emb_norm)
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard', scale=opts.scale, learnable_scale=opts.learnable_scale)
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2], attn_type=opts.attn_type, emb_norm=opts.emb_norm)
    elif opts.arch == 'adm':
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])
    elif 'edm2-img64' in opts.arch:
        nc_dict = {'s': 192, 'm': 256, 'l': 320, 'xl': 384}
        c.network_kwargs.update(model_type='EDM2UNet', model_channels=nc_dict[opts.arch.split('-')[-1]])
        c.network_kwargs.update(scale=opts.scale, emb_norm=opts.emb_norm, learnable_scale=opts.learnable_scale)
    elif opts.arch == 'edm2-cifar-m':
        c.network_kwargs.update(model_type='EDM2UNet', model_channels=128, attn_resolutions=[16], channel_mult=[2,2,2], num_blocks=4) # For cifar-10
        c.network_kwargs.update(scale=opts.scale, emb_norm=opts.emb_norm, learnable_scale=opts.learnable_scale)
    elif opts.arch == 'edm2-cifar-s':
        c.network_kwargs.update(model_type='EDM2UNet', model_channels=128, attn_resolutions=[16], channel_mult=[1,2,2,2]) # For cifar-10
        c.network_kwargs.update(scale=opts.scale, emb_norm=opts.emb_norm, learnable_scale=opts.learnable_scale)

    else:
        raise ValueError(f"Unrecognized architecture: {opts.arch}")

    # Preconditioning.
    c.network_kwargs.class_name = 'training.networks.ECMPrecond'

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Trainig options.
    c.update(cudnn_benchmark=opts.bench)
    if opts.mid_t is not None:
        opts.mid_t = [float(x) for x in opts.mid_t]
    c.update(mid_t=opts.mid_t, metrics=opts.metrics)
    c.update(im_dir=opts.im_dir, latent_dir=opts.latent_dir)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        c.seed = 0

    # Checkpoint to evaluate.
    c.resume_pkl = opts.resume
    c.save_pth = opts.save_pth

    # Number of samples to generate.
    c.num_samples = opts.num_samples
    c.class_label = opts.class_label

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)
        

    # Print options.
    dist.print0()
    dist.print0('Evaluarion options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    evaluation(**c)

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 16)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 16)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)
    
#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

@torch.no_grad()
def generator_fn(
    net, latents, class_labels=None, 
    t_max=80, mid_t=None, data=None
):
    # Time step discretization.
    mid_t = [] if mid_t is None else mid_t
    t_steps = torch.tensor([t_max]+list(mid_t), dtype=torch.float64, device=latents.device)

    # t_0 = T, t_N = 0
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Sampling steps 
    x = latents.to(torch.float64) * t_steps[0]
    # Save cuda arng state to ensure that this for loop does not affect the program
    rng_stage = torch.cuda.get_rng_state()
    if data is not None:
        x = x + data
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x = net(x, t_cur, class_labels).to(torch.float64)
        if t_next > 0:
            x = x + t_next * torch.randn_like(x) 
    torch.cuda.set_rng_state(rng_stage)
    return x

@torch.no_grad()
def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, t_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, data=None
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(t_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next
#----------------------------------------------------------------------------

from tqdm import tqdm

def evaluation(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    network_kwargs      = {},       # Options for model and preconditioning.
    batch_size          = None,      # Total batch size for one training iteration.
    seed                = 0,        # Global random seed.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    mid_t               = None,     # Intermediate t for few-step generation.
    metrics             = None,     # Metrics for evaluation.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    im_dir              = None,    # Path to the synthetic image directory, for distillation error evaluation.
    latent_dir          = None,    # Path to the noise directory, for distillation error evaluation.
    save_pth            = False,    # Save the .pth file.
    num_samples         = 50000,    # Number of samples to generate.
    class_label         = None,    # Specific class label to generate
):
    # Initialize.
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    if batch_size is None:
        # Default batch size
        batch_gpu = 251
    else:
        batch_gpu = batch_size // dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset

    # Construct network.
    dist.print0('Constructing network...')

    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    

    net.eval().requires_grad_(False).to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if resume_pkl.endswith('.pkl'):
            if dist.get_rank() != 0:
                torch.distributed.barrier() # rank 0 goes first
            with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
                data = pickle.load(f)
            if dist.get_rank() == 0:
                torch.distributed.barrier() # other ranks follow
            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=True) #False)
            del data # conserve memory
        elif resume_pkl.endswith('.pt') or resume_pkl.endswith('.pth'):
            if dist.get_rank() != 0:
                torch.distributed.barrier() # rank 0 goes first
            net.load_state_dict(torch.load(resume_pkl, map_location='cpu'), strict=True)
            if dist.get_rank() == 0:
                torch.distributed.barrier() # other ranks follow
    if save_pth:
        torch.save(net.state_dict(), os.path.join(run_dir, 'model.pth'))
    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
        

        
    # Few-step Evaluation.
    few_step_fn = functools.partial(generator_fn, mid_t=mid_t)





    dist.print0('Evaluating few-step generation...')

    # Generate num_samples images and save them to outdir
    cnt = 0
    batch_nsamples = num_samples // dist.get_world_size() + 1
    while cnt < batch_nsamples:
        os.makedirs(run_dir, exist_ok=True)
        z = torch.randn([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        
        if dataset_obj.has_labels:
            if class_label is not None:
                # Use the specified class label for all images in the batch
                labels = torch.full((z.shape[0],), class_label, dtype=torch.long, device=device)
            else:
                # Generate random labels as before
                labels = torch.randint(0, dataset_obj.label_dim, (z.shape[0],), device=device)
            
            label_onehot = torch.eye(dataset_obj.label_dim, device=device)[labels]
        else:
            label_onehot = None

        images = few_step_fn(net, z, label_onehot).cpu()
        for img in images:
            save_image(img * 0.5 + 0.5, os.path.join(run_dir, f'sample_{cnt + batch_nsamples * dist.get_rank():05d}.png'))
            cnt += 1

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
