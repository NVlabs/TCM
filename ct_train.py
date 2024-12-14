# ---------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from EDM2 and ECT codebase, which built upon EDM.
#
# Source:
# https://github.com/NVlabs/edm/blob/main/train.py (EDM)
# https://github.com/locuslab/ect/blob/main/ct_train.py (ECT)
# https://github.com/NVlabs/edm2/blob/main/train_edm2.py (EDM2)
#
# The license for these can be found in license/ directory.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------


import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import ct_training_loop as training_loop
import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.


#----------------------------------------------------------------------------
# Scripts for auto-resuming

def create_expt_folder_with_auto_resuming(output_dir, name):
    exp_dir = os.path.join(output_dir, name)
    checkpoint = None
    if os.path.exists(exp_dir):
        all_tags = os.listdir(exp_dir)
        all_existing_tags = [tag for tag in all_tags if tag.startswith('tag')]
        all_existing_tags.sort()
        all_existing_tags = all_existing_tags[::-1]
        for previous_tag in all_existing_tags:
            potential_ckpt = os.path.join(exp_dir, previous_tag, 'training-state-latest.pt')
            if os.path.exists(potential_ckpt):
                checkpoint = potential_ckpt
                if dist.get_rank() == 0:
                    print('auto-resuming ckpt found ' + potential_ckpt)
                break
        curr_tag = 'tag' + str(len(all_existing_tags)).zfill(2)
        exp_dir = os.path.join(exp_dir, curr_tag)  # output/name/tagxx
    else:
        exp_dir = os.path.join(exp_dir, 'tag00')  # output/name/tag00
    dist.synchronize()
    if dist.get_rank() == 0:
        os.makedirs(exp_dir)
    return exp_dir, checkpoint


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
@click.option('--precond',       help='Preconditioning & loss function', metavar='ect',             type=click.Choice(['ect']), default='ect', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--optim',         help='Name of Optimizer', metavar='Optimizer',                     type=str, default='Adam', show_default=True)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--beta1',         help='Adam beta1', metavar='FLOAT',                                type=click.FloatRange(min=0, max=1), default=0.9, show_default=True)
@click.option('--beta2',         help='Adam beta2', metavar='FLOAT',                                type=click.FloatRange(min=0, max=1), default=0.999, show_default=True)
@click.option('--eps',help='Adam epsilon', metavar='FLOAT',                             type=click.FloatRange(min=0), default=1e-8, show_default=True)
@click.option('--decay_iter',    help='Decay learning rate after this many iterations, 0: no decay', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--rampup_iter',   help='Rampup learning rate for this many iterations, 0: no rampup', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--lr_schedule_start_iter', help='When computing the current iteration for lr scheduling, start from this iteration', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)

@click.option('--ema_type',      help='Type of EMA', metavar='STR',                                 default='constant', show_default=True)
@click.option('--ema_beta',      help='EMA decay rate', metavar='FLOAT',                            type=click.FloatRange(min=0), default=0.9999, show_default=True)
@click.option('--ema_gamma',     help='EMA decay rate for the rampup period', metavar='FLOAT',      type=click.FloatRange(min=0), default=None, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0., show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Model Hyperparameters
@click.option('--mean',          help='P_mean of Log Normal Distribution', metavar='FLOAT',         type=click.FloatRange(), default=-1.1, show_default=True)
@click.option('--std',           help='P_std of Log Normal Distribution', metavar='FLOAT',          type=click.FloatRange(), default=2.0, show_default=True)
@click.option('--t_lower',       help='Lower bound of t', metavar='FLOAT',                          type=click.FloatRange(), default=0.002, show_default=True)
@click.option('--t_upper',       help='Upper bound of t', metavar='FLOAT',                          type=click.FloatRange(), default=80, show_default=True)
@click.option('--tdist',        help='Type of t distribution', metavar='STR',                       default='normal', show_default=True)
@click.option('--df',           help='Degrees of freedom for t distribution', metavar='FLOAT',      type=click.FloatRange(), default=2.0, show_default=True)

@click.option('--double',        help='How often to reduce dt', metavar='TICKS',                    type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--start_stage', help='Start stage', metavar='INT', type=click.IntRange(min=-1), default=0, show_default=True)
@click.option('--w_boundary',    help='Weight for boundary condition', metavar='FLOAT',            type=click.FloatRange(), default=1, show_default=True)
@click.option('-q',              help='Decay Factor', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=2.0, show_default=True)
@click.option('-c',              help='Constant c for Huber Loss', metavar='FLOAT',                 type=click.FloatRange(), default=0.0, show_default=True)
@click.option('-k',              help='Mapping fn hyperparams', metavar='FLOAT',                    type=click.FloatRange(), default=8.0, show_default=True)
@click.option('-b',              help='Mapping fn hyperparams', metavar='FLOAT',                    type=click.FloatRange(), default=1.0, show_default=True)
@click.option('--cut',           help='Cutoff value.', metavar='FLOAT',                             type=click.FloatRange(), default=4.0, show_default=True)
@click.option('--boundary_prob', help='With this probability, convert t to transition_t for boundary condition', metavar='FLOAT', type=click.FloatRange(), default=0.05, show_default=True)

@click.option('--weighting',     help='Weighting function', metavar='STR',   default='default', show_default=True)
@click.option('--ratio_limit',   help='Limit the ratio', metavar='FLOAT',                           type=click.FloatRange(min=0, max=1), default=0.999, show_default=True)
@click.option('--sqrt',          help='Take sqrt to the squared L2 loss', metavar='BOOL',                   type=bool, default=True, show_default=True)
@click.option('--gclip',        help='Gradient clipping', metavar='FLOAT',                          type=click.FloatRange(min=0), default=1000000., show_default=True)
# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.FloatRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--ckpt',          help='How often to save latest checkpoints', metavar='TICKS',      type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('--resume_tick',   help='Number of tick from previous training state', metavar='INT', type=int)
@click.option('-n', '--dry_run', help='Print training options and exit',                            is_flag=True)
@click.option('--use_wandb',     help='Use wandb or not',                                           type=bool, default=False, show_default=True)
@click.option('--distill',       help='Path to teacher net for distillation', metavar='PKL|URL',    type=str)
@click.option('--stage_max',     help='Maximum stage in step scheduling', metavar='INT',            type=click.IntRange(min=1), default=100, show_default=True)

# TCM
@click.option('--tcm_teacher_pkl', help='Path to teacher net (stage-1 model) for stage-2 training', metavar='PKL|URL', type=str)
@click.option('--tcm_transition_t', help='Transition time t prime', metavar='FLOAT', type=click.FloatRange(min=0), default=1., show_default=True)
# Evaluation
@click.option('--mid_t',         help='Sampler steps [default: 0.821]',                             multiple=True, default=[0.821])
@click.option('--metrics',       help='Comma-separated list or "none" [default: fid50k_full]',      type=CommaSeparatedList(), default='fid50k_full')
@click.option('--sample_every',  help='How often to sample imgs', metavar='TICKS',                  type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--eval_every',    help='How often to evaluate metrics', metavar='TICKS',             type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dfid_ts', help='Comma-separated list of t_max values for FID calculation', type=CommaSeparatedList(), default='1.,2.,3.,4.')
@click.option('--dout_resolutions', help='Resolutions at which to apply dropout', metavar='LIST', type=CommaSeparatedList(), default='16,8,4,2,1')


def main(**kwargs):
    """Train ECMs using the techniques described in the
    blog "Consistency Models Made Easy".
    """
    opts = dnnlib.EasyDict(kwargs)

    opts.tick = opts.batch * 0.1 # 1 tick = 100 iterations
    # current_date = datetime.datetime.now().strftime("%m.%d.%Y")
    # opts.outdir = os.path.join(opts.outdir, current_date)
    print(f"--------------------------------ticks = {opts.tick}--------------------------------")
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict(P_mean=opts.mean, P_std=opts.std, t_lower=opts.t_lower, t_upper=opts.t_upper, tdist=opts.tdist, df=opts.df, 
                                    q=opts.q, c=opts.c, k=opts.k, b=opts.b, cut=opts.cut, ratio_limit=opts.ratio_limit, sqrt=opts.sqrt, weighting=opts.weighting, boundary_prob=opts.boundary_prob)
    c.optimizer_kwargs = dnnlib.EasyDict(class_name=f'torch.optim.{opts.optim}', lr=opts.lr, betas=[opts.beta1, opts.beta2], eps=opts.eps)
    c.lr_kawrgs = dnnlib.EasyDict(decay_iter=opts.decay_iter, rampup_iter=opts.rampup_iter, start_iter=opts.lr_schedule_start_iter)
    if opts.ema_type == 'constant':
        assert opts.ema_beta is not None, 'ema_beta must be specified for constant EMA'
    if opts.ema_type == 'power':
        assert opts.ema_gamma is not None, 'ema_gamma must be specified for power EMA'
    c.ema_kwargs = dnnlib.EasyDict(ema_type=opts.ema_type, ema_beta=opts.ema_beta, ema_gamma=opts.ema_gamma)
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
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'adm':
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])
    elif 'edm2-img64' in opts.arch:
        nc_dict = {'s': 192, 'm': 256, 'l': 320, 'xl': 384}
        c.network_kwargs.update(model_type='EDM2UNet', model_channels=nc_dict[opts.arch.split('-')[-1]])
        c.network_kwargs.update(dout_resolutions=[int(res) for res in opts.dout_resolutions])
    else:
        raise ValueError(f"Unrecognized architecture: {opts.arch}")

    # Preconditioning & loss function.
    if opts.precond == 'ect':
        c.network_kwargs.class_name = 'training.networks.ECMPrecond'
        c.loss_kwargs.class_name = 'training.loss.ECMLoss'
    else:
        raise ValueError('Unrecognized Precond & Loss!')

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Progressive training options.
    if opts.tcm_teacher_pkl is not None:
        if opts.tcm_transition_t > opts.t_lower:
            raise ValueError(f"tcm_transition_t must be less than t_lower, but got {opts.tcm_transition_t} > {opts.t_lower}")
        c.tcm_kwargs = dnnlib.EasyDict(teacher_pkl=opts.tcm_teacher_pkl, transition_t=opts.tcm_transition_t)
    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump, ckpt_ticks=opts.ckpt, double_ticks=opts.double)
    c.update(mid_t=opts.mid_t, metrics=opts.metrics, sample_ticks=opts.sample_every, eval_ticks=opts.eval_every)
    c.update(gclip=opts.gclip, start_stage=opts.start_stage, w_boundary=opts.w_boundary)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        c.seed = 101

    # Description string.
    # cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    # dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = opts.desc
    # desc = (f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-{opts.optim:s}-{opts.lr:f}'
    #         f'-mstage{opts.stage_max}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}')
    # if opts.desc is not None:
    #     desc += f'-{opts.desc}'
    c.desc = desc

    # Pick output directory.
    if opts.nosubdir:
        c.run_dir = opts.outdir
        ckpt_auto_resume = None
    else:
        c.run_dir, ckpt_auto_resume = create_expt_folder_with_auto_resuming(opts.outdir, desc)

    if ckpt_auto_resume is None:
        # Transfer learning and resume from given checkpoints or training states.
        if opts.transfer is not None:
            if opts.resume is not None:
                raise click.ClickException('--transfer and --resume cannot be specified at the same time')
            c.resume_pkl = opts.transfer
            c.resume_tick = 0 if opts.resume_tick is None else opts.resume_tick
        elif opts.resume is not None:
            match = re.fullmatch(r'training-state-(\d+|latest).pt', os.path.basename(opts.resume))
            if not match or not os.path.isfile(opts.resume):
                raise click.ClickException(f'--resume must point to training-state-*.pt from a previous training run, but got {opts.resume}')
            c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
            c.resume_tick = 0 if opts.resume_tick is None else opts.resume_tick
            c.resume_state_dump = opts.resume
    else:
        # Auto resuming form the last training state
        match = re.fullmatch(r'training-state-(\d+|latest).pt', os.path.basename(ckpt_auto_resume))
        if not match or not os.path.isfile(ckpt_auto_resume):
            raise click.ClickException('Auto resuming must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(ckpt_auto_resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_state_dump = ckpt_auto_resume

    # teacher network path for distillation
    if opts.distill is not None:
        c.teacher_net_pkl = opts.distill

    c.stage_max = opts.stage_max
    c.use_wandb = opts.use_wandb

    c.dfid_ts = [round(float(t), 3) for t in opts.dfid_ts]

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Enable distillation:     {opts.distill is not None}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
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
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
