# ---------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from EDM2 and ECT codebase, which built upon EDM.
#
# Source:
# https://github.com/NVlabs/edm/blob/main/training/training_loop.py (EDM)
# https://github.com/locuslab/ect/blob/main/training/ct_training_loop.py (ECT)
# https://github.com/NVlabs/edm2/blob/main/training/training_loop.py (EDM2)
# https://github.com/NVlabs/edm2/blob/main/training/phema.py
#
# The license for these can be found in license/ directory.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import os
import time
import copy
import json
import pickle
import psutil
import functools
import PIL.Image
import numpy as np
import torch
import wandb
from scipy.stats import norm

import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

from metrics import metric_main
from ct_eval import generator_fn
from torch.nn.utils import clip_grad_norm_
import datetime
import matplotlib.pyplot as plt
# downsampling
import torch.nn.functional as TF

from .networks_tcm import TCMPrecond

#----------------------------------------------------------------------------
def power_function_beta(cur_iter, gamma):
    beta = (1 - 1/cur_iter) ** (gamma + 1)
    return beta

#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr, decay_iter, rampup_iter=0, start_iter=0):
    lr = ref_lr
    cur_iter = cur_nimg / batch_size - start_iter
    assert cur_iter >= 0, f"cur_iter: {cur_iter}, cur_nimg: {cur_nimg}, batch_size: {batch_size}, start_iter: {start_iter}"
    if decay_iter > 0:
        lr /= np.sqrt(max(cur_iter / decay_iter, 1))
        # lr /= np.sqrt(max(cur_nimg / (decay_iter * batch_size), 1))
    if rampup_iter > 0:
        lr *= min(cur_iter / rampup_iter, 1)
        # lr *= min(cur_nimg / (rampup_iter * batch_size), 1)
    return lr

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



def save_ckpt(run_dir, ckpt_id, ema, loss_fn, augment_pipe, dataset_kwargs):
    data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
    for key, value in data.items():
        if isinstance(value, torch.nn.Module):
            value = copy.deepcopy(value).eval().requires_grad_(False)
            misc.check_ddp_consistency(value)
            data[key] = value.cpu()
        del value  # conserve memory
    save_path = os.path.join(run_dir, f'network-snapshot-{ckpt_id}.pkl')
    if dist.get_rank() == 0:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    return save_path


def load_ckpt(resume_pkl, ema=None, net=None):
    if dist.get_rank() != 0:
        dist.synchronize()  # rank 0 goes first
    with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
        data = pickle.load(f)
    if dist.get_rank() == 0:
        dist.synchronize()  # other ranks follow
    if net is not None:
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
    if ema is not None:
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
    return ema, net

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    lr_kawrgs           = {},       # Options for learning rate.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    tcm_kwargs           = None,     # Options for progressive training, None = disable.
    ema_kwargs          = {'ema_type': 'constant', 'ema_beta': 0.9999, 'gamma': None}, # Options for EMA.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    lr_rampup_kimg      = 0,        # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 500,      # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    ckpt_ticks          = 100,      # How often to save latest checkpoints, None = disable.
    sample_ticks        = 50,       # How often to sample images, None = disable.
    eval_ticks          = 500,      # How often to evaluate models, None = disable.
    grad_ticks          = 500,      # How often to log gradient norms, None = disable.
    double_ticks        = 500,      # How often to evaluate models, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_tick         = 0,        # Start from the given training progress.
    mid_t               = None,     # Intermediate t for few-step generation.
    metrics             = None,     # Metrics for evaluation.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    use_wandb           = False,    # Use wandb?
    desc                = 'ect',    # Description string for the project.
    teacher_net_pkl     = None,     # Path to the teacher network for consistency distillation.
    stage_max           = 100,      # Maximum stage for step scheduling.
    device              = torch.device('cuda'),
    gclip               = 1000000,     # Gradient clipping
    start_stage        = 0,        # -1: ratio starts at 0, 0: ratio starts at 1-1/q, 1: ratio starts at 1-1/q^2, and so on.
    w_boundary         = 10,
    dfid_ts            = None,     # List of t_max values for denoising FID calculation
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Rng states for evaluation.
    eval_rng_state = torch.get_rng_state()
    eval_cuda_rng_state = torch.cuda.get_rng_state()
    eval_np_rng_state = np.random.get_state()

    if dist.get_rank() == 0:
        config_path = os.path.join(run_dir, 'training_options.json')
        current_date = datetime.datetime.now().strftime("%m.%d.%Y")

        # Assuming 'desc' is predefined somewhere in your code
        with open(config_path, 'r') as f:
            config = json.load(f)
        if use_wandb:
            wandb.init(
                        project='project_name',
                        entity='user/team name',
                       name=desc,
                       group='exp_group_name',
                       reinit=True,
                       config=config,
                       settings=wandb.Settings(start_method='fork'))
            wandb.run.log_code(".")

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs).to(device) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    
    enable_gradscaler = loss_scaling != 1
    dist.print0(f'GradScaler enabled: {enable_gradscaler}')
    grad_scaler_scale = None
    if enable_gradscaler:
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-gradscaler
        # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
        dist.print0(f'Setting up GradScaler...')
        scaler = torch.cuda.amp.GradScaler(init_scale=loss_scaling, growth_interval=20000)
        dist.print0(f'Loss scaling is overwritten when GradScaler is enabled')
        grad_scaler_scale = loss_scaling
    dist.print0('Setting up DDP...')
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # --------------------------------------------------------------------------
    dist.synchronize()
    
    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        # if .pkl: load network weights, if .pt: load training state
        if resume_pkl.endswith('.pkl'):
            ema, net = load_ckpt(resume_pkl, ema, net)
        elif resume_pkl.endswith('.pt') or resume_pkl.endswith('.pth'):
            data = torch.load(resume_pkl, map_location=torch.device('cpu'))
            net.load_state_dict(data, strict=True)
            ema.load_state_dict(data, strict=True)
            del data
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        if enable_gradscaler:
            if 'gradscaler_state' in data:
                dist.print0(f'Loading GradScaler state from "{resume_state_dump}"...')
                # Although not loading the state_dict of the GradScaler works well, loading it can improve reproducibility.
                scaler.load_state_dict(data['gradscaler_state'])
            else:
                dist.print0(f'GradScaler state is not found in "{resume_state_dump}", using the default state.')
        resume_tick = data['resume_tick']
        del data # conserve memory
    else:
        resume_tick = resume_tick
    teacher_net = None
    if teacher_net_pkl is not None:
        teacher_net = copy.deepcopy(net).eval().requires_grad_(False)
        dist.print0(f'Loading teacher network weights from "{teacher_net_pkl}"...')
        teacher_net, _ = load_ckpt(teacher_net_pkl, teacher_net, net=None)

    if tcm_kwargs is not None:
        net_t = copy.deepcopy(net).requires_grad_(False)
        dist.print0('Constructing teacher network for stage-2...')
        if tcm_kwargs['teacher_pkl'].endswith('.pkl'):
            net_t, _ = load_ckpt(tcm_kwargs['teacher_pkl'], net_t, net=None)
        elif tcm_kwargs['teacher_pkl'].endswith('.pt') or tcm_kwargs['teacher_pkl'].endswith('.pth'):
            data = torch.load(tcm_kwargs['teacher_pkl'], map_location=torch.device('cpu'))
            net_t.load_state_dict(data, strict=True)
            del data
        ddp_tcm = TCMPrecond(net_t, ddp, **tcm_kwargs)
    else:
        ddp_tcm = None

    dist.print0(f'resume_tick: {resume_tick}')
    
    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
        
    if dist.get_rank() == 0:
        dist.print0('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=dataset_obj)
        save_image_grid(images, os.path.join(run_dir, 'data.png'), drange=[0,255], grid_size=grid_size)
        
        grid_z = torch.randn([labels.shape[0], ema.img_channels, ema.img_resolution, ema.img_resolution], device=device)
        grid_z = grid_z.split(batch_gpu)
        
        grid_c = torch.from_numpy(labels).to(device)
        grid_c = grid_c.split(batch_gpu)
        
        images = [generator_fn(ema, z, c).cpu() for z, c in zip(grid_z, grid_c)]
        images = torch.cat(images).numpy()
        save_image_grid(images, os.path.join(run_dir, 'model_init.png'), drange=[-1,1], grid_size=grid_size)
        del images

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0(f"ema_kwargs: {ema_kwargs}")
    dist.print0()
    cur_nimg = resume_tick * kimg_per_tick * 1000
    cur_tick = resume_tick
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg / 1000, total_kimg)
    stats_jsonl = None

    # Prepare for the mapping fn p(r|t).
    dist.print0(f'Reduce dt every {double_ticks} ticks.')
    
    def update_scheduler(loss_fn):
        cur_stage = min(stage, stage_max)
        loss_fn.update_schedule(cur_stage)
        dist.print0(f'Update scheduler at {cur_tick} ticks, {cur_nimg / 1e3} kimg, ratio {loss_fn.ratio}')
    stage = cur_tick // double_ticks + start_stage

    update_scheduler(loss_fn)
    
    logger = misc.LoggingT(n_bins=100)
    logger_unweighted = misc.LoggingT(n_bins=100)

    image_test = None
    grad_norm_list = None
    save_t = True
    while True:
   
        
        # Accumulate gradients.
        loss_batch = 0
        loss_unweighted_batch = 0
        loss_boundary_batch = 0
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)

                if image_test is None:
                    image_test = images
                    labels_test = labels

                loss_unweighted, loss, t, l2_distance, loss_boundary = loss_fn(net=ddp if ddp_tcm is None else ddp_tcm, images=images, labels=labels, augment_pipe=augment_pipe, teacher_net=teacher_net)
                
                if save_t and dist.get_rank() == 0:
                    # Save histogram of ln(t) to opts.run_dir/t_hist_i{cur_tick}.jpg
                    plt.hist(torch.log(t).cpu().numpy().flatten(), bins=100)
                    plt.xlabel('ln(t)')
                    plt.title(f"t histogram at cur_tick: {cur_tick}")
                    plt.savefig(os.path.join(run_dir, f"t_hist_i{cur_tick}.jpg"))
                    plt.close()
                    save_t = False

                

                logger.update(t, loss)
                logger_unweighted.update(t, loss_unweighted)            
                training_stats.report('Loss/loss', loss)
                if enable_gradscaler:
                    scaler.scale(loss.mean() + loss_boundary.mean() * w_boundary).backward()
                else:
                    (loss.mean() + loss_boundary.mean() * w_boundary).mul(loss_scaling).backward()

                loss_batch += loss.mean().item()
                loss_unweighted_batch += loss_unweighted.mean().item()
                loss_boundary_batch  += loss_boundary.mean().item()
        dist.print0(f"cur_tick: {cur_tick}, loss_batch: {loss_batch}, loss_unweighted_batch: {loss_unweighted_batch}, loss_boundary_batch: {loss_boundary_batch}")

        if enable_gradscaler:
            scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(net.parameters(), gclip).float().item()
        

        # Learning rate decay
        lr = learning_rate_schedule(cur_nimg, batch_size, ref_lr = optimizer_kwargs['lr'], decay_iter = lr_kawrgs['decay_iter'], rampup_iter = lr_kawrgs['rampup_iter'], start_iter = lr_kawrgs['start_iter'])
        for g in optimizer.param_groups:
            g['lr'] = lr

        if enable_gradscaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            
            optimizer.step()

        # Update EMA.
        cur_nimg += batch_size
        iterations = cur_nimg // batch_size

        if ema_kwargs['ema_type'] == 'constant':
            ema_beta = ema_kwargs['ema_beta']
        elif ema_kwargs['ema_type'] == 'power':
            ema_beta = power_function_beta(cur_iter=iterations, gamma=ema_kwargs['ema_gamma'])
        else:
            raise ValueError(f"Unknown EMA type: {ema_kwargs['ema_type']}")
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != resume_tick) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        if (eval_ticks is not None) and (done or cur_tick % eval_ticks == 0 or cur_tick == resume_tick):
            dist.print0('Evaluating models...')
            # Save current rng states
            main_rng_state = torch.get_rng_state()
            main_cuda_rng_state = torch.cuda.get_rng_state()
            main_np_rng_state = np.random.get_state()

            # Set rng states for evaluation
            torch.set_rng_state(eval_rng_state)
            torch.cuda.set_rng_state(eval_cuda_rng_state)
            np.random.set_state(eval_np_rng_state)
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, 
                    generator_fn=generator_fn, G=ema, G_kwargs={},
                    dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device, dfid_ts=dfid_ts)
                if dist.get_rank() == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'network-snapshot-{cur_tick:06d}.pkl')
                    if use_wandb:
                        wandb.log(result_dict['results'], step=cur_tick)
            # Restore rng states
            torch.set_rng_state(main_rng_state)
            torch.cuda.set_rng_state(main_cuda_rng_state)
            np.random.set_state(main_np_rng_state)

        
        if dist.get_rank() == 0 and use_wandb:
            # Save the weighted loss plot
            weighted_plot_path = os.path.join(run_dir, f"loss_i{cur_tick}.jpg")
            logger.plot(weighted_plot_path)


            # Save the unweighted loss plot
            unweighted_plot_path = os.path.join(run_dir, f"loss_i{cur_tick}_unweighted.jpg")
            logger_unweighted.plot(unweighted_plot_path)

            dist.print0(f"weighted_plot_path: {weighted_plot_path} saved")
            dist.print0(f"unweighted_plot_path: {unweighted_plot_path} saved")
            
            
            # Log the saved plots to wandb
            log_dict = {
                'train loss': loss_batch, 
                'train loss unweighted': loss_unweighted_batch, 
                'loss_boundary': loss_boundary_batch,
                'kimg': cur_nimg / 1e3, 
                'ratio': loss_fn.ratio, 
                "weighted_loss_plot": wandb.Image(weighted_plot_path),
                "unweighted_loss_plot": wandb.Image(unweighted_plot_path),
                "grad_norm": grad_norm, 
                "lr": lr
            }
            
            # Add GradScaler scale if it's enabled
            if enable_gradscaler:
                grad_scaler_scale = scaler.get_scale()
                log_dict['grad_scaler_scale'] = grad_scaler_scale
            
            
            wandb.log(log_dict, step=cur_tick)

            print(f"lr: {lr}")


        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"iteration {iterations}"]
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"loss {training_stats.default_collector['Loss/loss']:<9.5f}"]
        fields += [f"loss_unweighted {loss_unweighted_batch:<9.5f}"]
        fields += [f"ratio {loss_fn.ratio:<9.10f}"]
        fields += [f"grad_norm {grad_norm:<9.5f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        fields += [f"gpuutil {training_stats.report0('Resources/gpu_utilization', torch.cuda.utilization(device)):<6.2f}"]
        if grad_scaler_scale is not None:
            fields += [f"grad_scaler_scale {grad_scaler_scale:<9.5f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))


        dist.print0(f"cur_tick: {cur_tick}")
        
        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0) and cur_tick != 0:
            save_ckpt(run_dir, f'{cur_tick:06d}', ema, loss_fn, augment_pipe, dataset_kwargs)

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            if enable_gradscaler:
                data = dict(net=net,
                            optimizer_state=optimizer.state_dict(),
                            resume_tick=cur_tick,
                            gradscaler_state=scaler.state_dict(),
                            )
            else:
                data = dict(net=net,
                            optimizer_state=optimizer.state_dict(),
                            resume_tick=cur_tick,
                            )
            torch.save(data, os.path.join(run_dir, f'training-state-{cur_tick:06d}.pt'))           
           

        # Save latest checkpoints
        if (ckpt_ticks is not None) and (done or cur_tick % ckpt_ticks == 0) and cur_tick != 0:
            dist.print0(f'Save the latest checkpoint at {cur_tick:06d} ticks...')
            save_ckpt(run_dir, 'latest', ema, loss_fn, augment_pipe, dataset_kwargs)
            if dist.get_rank() == 0:
                if enable_gradscaler:
                    data = dict(net=net,
                                optimizer_state=optimizer.state_dict(),
                                gradscaler_state=scaler.state_dict(),
                                resume_tick=cur_tick,
                                )
                else:
                    data = dict(net=net,
                                optimizer_state=optimizer.state_dict(),
                                resume_tick=cur_tick,
                                )
                torch.save(data, os.path.join(run_dir, f'training-state-latest.pt'))

        # Sample Img
        if (sample_ticks is not None) and (done or cur_tick % sample_ticks == 0) and dist.get_rank() == 0:
            dist.print0('Exporting sample images...')
            images = [generator_fn(ema, z, c).cpu() for z, c in zip(grid_z, grid_c)]
            images = torch.cat(images).numpy()
            save_image_grid(images, os.path.join(run_dir, f'{cur_tick:06d}.png'), drange=[-1,1], grid_size=grid_size)
            del images
    
        

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_dict = dict(training_stats.default_collector.as_dict(), timestamp=time.time())
            if grad_scaler_scale is not None:
                stats_dict['grad_scaler_scale'] = grad_scaler_scale
            stats_jsonl.write(json.dumps(stats_dict) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg / 1000, total_kimg)

        # --------------------------------------------------------------------------
        

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

        # Update Scheduler
        new_stage = (cur_tick - 1) // double_ticks + start_stage
        if new_stage > stage:
            stage = new_stage
            update_scheduler(loss_fn)

    # Few-step Evaluation.
    few_step_fn = functools.partial(generator_fn, mid_t=mid_t)
    
    if dist.get_rank() == 0:
        dist.print0('Exporting final sample images...')
        images = [few_step_fn(ema, z, c).cpu() for z, c in zip(grid_z, grid_c)]
        images = torch.cat(images).numpy()
        save_image_grid(images, os.path.join(run_dir, 'final.png'), drange=[-1,1], grid_size=grid_size)
        del images

    dist.print0('Evaluating few-step generation...')
    for _ in range(1):
        for metric in metrics:
            result_dict = metric_main.calc_metric(metric=metric, 
                generator_fn=few_step_fn, G=ema, G_kwargs={},
                dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
            if dist.get_rank() == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl='network-snapshot-latest.pkl')

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------