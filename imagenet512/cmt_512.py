import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import cmt_training_loop_simpleema_512 as training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.


#----------------------------------------------------------------------------
# Configuration presets.

config_presets = {
    'edm2-img512-s':    dnnlib.EasyDict(channels=192, dropout=0.0),
    'edm2-img512-m':    dnnlib.EasyDict(channels=256, dropout=0.0),
    'edm2-img512-l':    dnnlib.EasyDict(channels=320, dropout=0.0),
    'edm2-img512-xxl':  dnnlib.EasyDict(channels=448, dropout=0.0),
}

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
# @click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm|edm2',     type=click.Choice(['ddpmpp', 'ncsnpp', 'adm', 'edm2']), default='edm2', show_default=True)
@click.option('--preset',        help='Configuration preset', metavar='STR',                        type=str, default='', show_default=True)
@click.option('--scheme',        help='Learning scheme', metavar='ect',                             type=click.Choice(['ect']), default='ect', show_default=True)
@click.option('--wt',            help='Weighting function', metavar='snrpk',                        type=click.Choice(['snrpk']), default='snrpk', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--optim',         help='Name of Optimizer', metavar='Optimizer',                     type=str, default='Adam', show_default=True)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--sch',           help='Use scheduler or not', metavar='INT',                        type=int, default=1, show_default=True)
@click.option('--decay',         help='Learning rate decat', metavar='int',                         type=click.FloatRange(min=0, min_open=False), default=0, show_default=True)
@click.option('--betas',         help='betas for optimizer', metavar='FLOAT',                       multiple=True, default=(0.9, 0.999), show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=None, show_default=True)
@click.option('--ema_beta',      help='EMA decay rate', metavar='FLOAT',                            type=click.FloatRange(min=0), default=0.9999, show_default=True)
@click.option('--dropres',       help='Feature Size where dropout applied', metavar='FLOAT',        type=click.FloatRange(min=0), default=None, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)

# Model Hyperparameters
@click.option('--nfe',           help='NFE in DPM-Solver', metavar='INT',                           type=click.IntRange(min=1), default=16, show_default=True)
@click.option('--loss_metric',   help='Loss mertric type', metavar='STR',                           type=str, default='MSE', show_default=True)
@click.option('--loss_c',        help='c in Loss', metavar='FLOAT',                                 type=click.FloatRange(min=0, max=1), default=0.06)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
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
@click.option('--resume-tick',   help='Number of tick from previous training state', metavar='INT', type=int)
@click.option('-n', '--dry_run', help='Print training options and exit',                            is_flag=True)

# Evaluation
@click.option('--mid_t',         help='Sampler steps [default: 0.776]', metavar='FLOAT',            multiple=True, default=(7.7569e-01,), show_default=True)
@click.option('--metrics',       help='Comma-separated list or "none" [default: fid50k_full]',      type=CommaSeparatedList(), default='fid50k_full', show_default=True)
@click.option('--sample_every',  help='How often to sample imgs', metavar='TICKS',                  type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--eval_every',    help='How often to evaluate metrics', metavar='TICKS',             type=click.IntRange(min=1), default=500, show_default=True)

def main(**kwargs):
    """Train ECMs using the techniques described in "Consistency Models Made Easy".
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Preset.
    if opts.preset:
        if opts.preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{opts.preset}"')
        for key, value in config_presets[opts.preset].items():
            opts[key] = value
    
    dist.print0(opts)

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(use_labels=opts.cond)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.loss_kwargs = dnnlib.EasyDict(NFE=opts.nfe, loss_metric=opts.loss_metric, c=opts.loss_c)
    c.optimizer_kwargs = dnnlib.EasyDict(class_name=f'torch.optim.{opts.optim}', lr=opts.lr, betas=opts.betas, eps=1e-8)

    # Validate dataset options.
    try:
        dataset_name = 'imgnet512'
        c.dataset_kwargs.resolution = 64 # be explicit about dataset resolution
        c.dataset_kwargs.max_size = 2560 # be explicit about dataset size
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StabilityVAEEncoder')
    
    # Network architecture.
    assert opts.arch == 'edm2'
    c.network_kwargs = dnnlib.EasyDict(class_name='training.networks_edm2.Precond', model_channels=opts.channels)
    c.ref_network_kwargs = dnnlib.EasyDict(class_name='training.networks_edm2.Precond', model_channels=opts.channels)
    
    # Preconditioning & loss function.
    c.loss_kwargs.class_name = 'training.loss.DPMSolverLoss'

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
        c.ref_network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
        c.ref_network_kwargs.channel_mult = opts.cres
    c.network_kwargs.update(dropout=opts.dropout, dropout_res=opts.dropres, use_fp16=opts.fp16)
    c.ref_network_kwargs.update(dropout=opts.dropout, dropout_res=opts.dropres, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000) if opts.ema is not None else opts.ema
    c.ema_beta = opts.ema_beta
    c.sch = opts.sch
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump, ckpt_ticks=opts.ckpt)
    c.update(mid_t=opts.mid_t, metrics=opts.metrics, sample_ticks=opts.sample_every, eval_ticks=opts.eval_every)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Load checkpoints.
    if opts.transfer is not None:
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
        # NOTE: test resume-tick here
        c.resume_tick = 0 if opts.resume_tick is None else opts.resume_tick
    if opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+|latest)(?:-(\d+))?\.pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        # c.resume_pkl = None
        c.resume_tick = int(match.group(1)) if match.group(1).isdigit() else opts.resume_tick
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.preset:s}-{opts.optim:s}-{opts.lr:f}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
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
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Loss:                    {opts.scheme}')
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
