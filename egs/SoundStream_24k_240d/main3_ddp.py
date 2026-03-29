# 与 Encodec_24k_240d main3_ddp.py 相比只有鉴别器不同
import argparse
import glob
import itertools
import os
import time

import torch
import geoopt
import torch.distributed as dist
from academicodec.models.encodec.distributed.launch import launch
from academicodec.models.encodec.msstftd import MultiScaleSTFTDiscriminator
from academicodec.models.encodec.net3 import SoundStream
from academicodec.models.soundstream.dataset import NSynthDataset
from academicodec.models.soundstream.loss import criterion_d
from academicodec.models.soundstream.loss import criterion_g
from academicodec.models.soundstream.loss import loss_dis
from academicodec.models.soundstream.loss import loss_g
from academicodec.models.soundstream.models import MultiPeriodDiscriminator
from academicodec.models.soundstream.models import MultiScaleDiscriminator
from academicodec.utils import Logger
from academicodec.utils import seed_everything
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
NODE_RANK = os.environ['INDEX'] if 'INDEX' in os.environ else 0
NODE_RANK = int(NODE_RANK)
MASTER_ADDR, MASTER_PORT = (os.environ['CHIEF_IP'],
                            22275) if 'CHIEF_IP' in os.environ else (
                                "127.0.0.1", 29500)
MASTER_PORT = int(MASTER_PORT)
DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)
NUM_NODE = os.environ['HOST_NUM'] if 'HOST_NUM' in os.environ else 1


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


def _raise_non_finite(name, tensor, stage):
    finite_mask = torch.isfinite(tensor)
    bad_idx = (~finite_mask).nonzero(as_tuple=False)
    first_bad = bad_idx[0].tolist() if bad_idx.numel() > 0 else []
    nan_count = torch.isnan(tensor).sum().item()
    posinf_count = torch.isposinf(tensor).sum().item()
    neginf_count = torch.isneginf(tensor).sum().item()
    finite_vals = tensor[finite_mask]
    finite_min = finite_vals.min().item() if finite_vals.numel() > 0 else float('nan')
    finite_max = finite_vals.max().item() if finite_vals.numel() > 0 else float('nan')
    raise RuntimeError(
        f"Non-finite detected at {stage}: {name}, shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, nan={nan_count}, +inf={posinf_count}, -inf={neginf_count}, "
        f"first_bad_index={first_bad}, finite_min={finite_min:.6g}, finite_max={finite_max:.6g}"
    )


def check_quantizer_finite(module, stage):
    for name, param in module.named_parameters():
        if 'quantizer' not in name and 'codebook' not in name:
            continue
        if not torch.isfinite(param.data).all():
            _raise_non_finite(name, param.data, stage)
        if param.grad is not None and not torch.isfinite(param.grad).all():
            _raise_non_finite(f"{name}.grad", param.grad, stage)


def check_quantizer_optimizer_state_finite(module, optimizer, stage):
    tracked = {
        param: name for name, param in module.named_parameters()
        if ('quantizer' in name or 'codebook' in name)
    }
    for param, state in optimizer.state.items():
        name = tracked.get(param)
        if name is None:
            continue
        for state_key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            tensor = state.get(state_key)
            if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
                _raise_non_finite(f"optimizer_state[{state_key}]/{name}", tensor, stage)


def clip_named_grad_norm(module, max_norm, *, include_quantizer=True, manifold_only=False):
    params = []
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        if include_quantizer and ('quantizer' not in name and 'codebook' not in name):
            continue
        if manifold_only and not hasattr(param, 'manifold'):
            continue
        params.append(param)
    if len(params) == 0:
        return 0.0
    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    if torch.is_tensor(total_norm):
        total_norm = total_norm.item()
    return float(total_norm)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_node',
        type=int,
        default=NUM_NODE,
        help='number of nodes for distributed training')
    parser.add_argument(
        '--ngpus_per_node',
        type=int,
        default=8,
        help='number of gpu on one node')
    parser.add_argument(
        '--node_rank',
        type=int,
        default=NODE_RANK,
        help='node rank for distributed training')
    parser.add_argument(
        '--dist_url',
        type=str,
        default=DIST_URL,
        help='url used to set up distributed training')
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU id to use. If given, only the specific gpu will be'
        ' used, and ddp will be disabled')
    parser.add_argument(
        '--local_rank',
        default=-1,
        type=int,
        help='node rank for distributed training')
    # args for random
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='seed for initializing training. ')
    parser.add_argument(
        '--cudnn_deterministic',
        action='store_true',
        help='set cudnn.deterministic True')
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='use tensorboard for logging')
    # args for training
    parser.add_argument(
        '--LAMBDA_ADV',
        type=float,
        default=1,
        help='hyper-parameter for adver loss')
    parser.add_argument(
        '--LAMBDA_FEAT',
        type=float,
        default=1,
        help='hyper-parameter for feat loss')
    parser.add_argument(
        '--LAMBDA_REC',
        type=float,
        default=1,
        help='hyper-parameter for rec loss')
    parser.add_argument(
        '--LAMBDA_COM',
        type=float,
        default=1000,
        help='hyper-parameter for commit loss')
    parser.add_argument(
        '--N_EPOCHS', type=int, default=100, help='Total training epoch')
    parser.add_argument(
        '--st_epoch', type=int, default=1, help='start training epoch')
    parser.add_argument(
        '--global_step', type=int, default=0, help='record the global step')
    parser.add_argument('--discriminator_iter_start', type=int, default=500)
    parser.add_argument('--BATCH_SIZE', type=int, default=2, help='batch size')
    parser.add_argument(
        '--PATH',
        type=str,
        default='model_path/',
        help='The path to save the model')
    parser.add_argument('--sr', type=int, default=24000, help='sample rate')
    parser.add_argument(
        '--print_freq', type=int, default=10, help='the print number')
    # --save_dir kept for backward compat but defaults to PATH
    parser.add_argument(
        '--save_dir', type=str, default=None, help='(deprecated, uses PATH)')
    parser.add_argument(
        '--train_data_path',
        type=str,
        default='path_to_wavs',
        help='training data')
    parser.add_argument(
        '--valid_data_path',
        type=str,
        default='path_to_val_wavs',
        help='training data')
    parser.add_argument(
        '--resume', action='store_true', help='whether re-train model')
    parser.add_argument(
        '--resume_path', type=str, default='path_to_resume', help='resume_path')
    parser.add_argument(
        '--c', type=float, default=0.0, help='hyper-parameter for hyperbolic space')
    parser.add_argument(
        '--ratios',
        type=int,
        nargs='+',
        # probs(ratios) = hop_size
        default=[8, 5, 4, 2],
        help='ratios of SoundStream, shoud be set for different hop_size (32d, 320, 240d, ...)'
    )
    parser.add_argument(
        '--target_bandwidths',
        type=float,
        nargs='+',
        # default for 16k_320d
        default=[1, 1.5, 2, 4, 6, 12],
        help='target_bandwidths of net3.py')
    parser.add_argument(
        '--ema',
        action='store_true',
        help='use EMA for codebook (default: False)')
    parser.add_argument(
        '--kmeans_init',
        action='store_true',
        help='use kmeans_init for codebook (default: False)')
    parser.add_argument(
        '--pre_quant_batchnorm',
        action='store_true',
        help='apply BatchNorm1d on encoder output right before quantization')
    parser.add_argument(
        '--codebook_number',
        type=int,
        default=0,
        help='which codebook to visualize (default: 0)')
    parser.add_argument(
        '--number_of_steps',
        type=int,
        default=500,
        help='save codebook every number_of_steps batches (default: 100)')
    parser.add_argument(
        '--lr_g',
        type=float,
        default=3e-4,
        help='base learning rate for generator and euclidean parameters')
    parser.add_argument(
        '--lr_manifold',
        type=float,
        default=1e-5,
        help='learning rate for manifold parameters (geoopt)')
    parser.add_argument(
        '--geoopt_eps',
        type=float,
        default=1e-5,
        help='epsilon for geoopt RiemannianAdam denominator stability')
    parser.add_argument(
        '--geoopt_stabilize',
        type=int,
        default=1,
        help='apply manifold stabilization every N steps (1 = every step)')
    parser.add_argument(
        '--quantizer_grad_clip',
        type=float,
        default=0.05,
        help='max norm for quantizer/codebook gradients before global clipping')
    parser.add_argument(
        '--manifold_grad_clip',
        type=float,
        default=0.02,
        help='max norm for manifold quantizer gradients before optimizer step')
    parser.add_argument(
        '--warmup_epochs_g',
        type=int,
        default=0,
        help='number of linear warmup epochs for generator LR scheduler')
    args = parser.parse_args()
    if 'SLURM_JOB_ID' in os.environ:
        time_str = os.environ['SLURM_JOB_ID']
    else:
        time_str = time.strftime('%Y-%m-%d-%H-%M')
    
    if args.resume:
        args.PATH = args.resume_path  # direcly use the old model path
    else:
        args.PATH = os.path.join(args.PATH, time_str)
    # Unify save_dir into PATH (like MNIST VQ-VAE)
    args.save_dir = args.PATH
    os.makedirs(args.PATH, exist_ok=True)
    return args


def get_input(x):
    x = x.to(memory_format=torch.contiguous_format)
    return x.float()


def main():
    args = get_args()
    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)
    if args.num_node == 1:
        args.dist_url == "auto"
    else:
        assert args.num_node > 1
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.num_node  #
    launch(
        main_worker,
        args.ngpus_per_node,
        args.num_node,
        args.node_rank,
        args.dist_url,
        args=(args, ))


def add_nan_hook(model, model_name):
    def check_nan(name, x):
        if torch.is_tensor(x) and torch.isnan(x).any():
            print(f"NaN DETECTED in {name}", flush=True)
            import sys
            sys.exit(1)
            
    def hook(module, input, output):
        module_name = f"{model_name}.{module.__class__.__name__}"
        if isinstance(output, tuple):
            for i, out in enumerate(output):
                check_nan(f"{module_name} (output {i})", out)
        elif isinstance(output, list):
            for i, out in enumerate(output):
                check_nan(f"{module_name} (output {i})", out)
        elif isinstance(output, dict):
            for k, out in output.items():
                check_nan(f"{module_name} (output {k})", out)
        else:
            check_nan(module_name, output)

    for name, submodule in model.named_modules():
        submodule.register_forward_hook(hook)

def main_worker(local_rank, args):
    args.local_rank = local_rank
    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1
    # torch.autograd.set_detect_anomaly(True)
    #CUDA_VISIBLE_DEVICES = int(args.local_rank)
    logger = Logger(args)
    # 240倍下采
    soundstream = SoundStream(n_filters=32, D=512, ratios=args.ratios, c=args.c,
                              ema=args.ema, kmeans_init=args.kmeans_init,
                              pre_quant_batchnorm=args.pre_quant_batchnorm)
    #print(soundstream)
    msd = MultiScaleDiscriminator()
    mpd = MultiPeriodDiscriminator()
    stft_disc = MultiScaleSTFTDiscriminator(filters=32)
    getModelSize(soundstream)
    getModelSize(msd)
    getModelSize(mpd)
    getModelSize(stft_disc)
    if args.distributed:
        soundstream = torch.nn.SyncBatchNorm.convert_sync_batchnorm(soundstream)
        stft_disc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(stft_disc)
        msd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(msd)
        mpd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(mpd)
    # torch.distributed.barrier()
    args.device = torch.device('cuda', args.local_rank)
    soundstream.to(args.device)
    stft_disc.to(args.device)
    msd.to(args.device)
    mpd.to(args.device)
    if args.distributed:
        soundstream = DDP(
            soundstream,
            device_ids=[args.local_rank],
            find_unused_parameters=True
        )  # device_ids=[args.local_rank], output_device=args.local_rank
        stft_disc = DDP(stft_disc,
                        device_ids=[args.local_rank],
                        find_unused_parameters=True)
        msd = DDP(msd,
                  device_ids=[args.local_rank],
                  find_unused_parameters=True)
        mpd = DDP(mpd,
                  device_ids=[args.local_rank],
                  find_unused_parameters=True)

    train_dataset = NSynthDataset(audio_dir=args.train_data_path)
    valid_dataset = NSynthDataset(audio_dir=args.valid_data_path)
    args.sr = train_dataset.sr
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, drop_last=True, shuffle=True)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset)
    else:
        train_sampler = None
        valid_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=8,
        sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=8,
        sampler=valid_sampler)
        
    # manifold_params = []
    # euclidean_params = []

    # for p in soundstream.parameters():
    #     if hasattr(p, "manifold"):
    #         manifold_params.append(p)
    #     else:
    #         euclidean_params.append(p)

    # print(f"Manifold params: {len(manifold_params)}")

    # if args.c > 0:
    #     optimizer_g = geoopt.optim.RiemannianAdam(
    #         manifold_params,
    #         lr=3e-4,
    #         betas=(0.5, 0.9),
    #         eps=1e-8
    #     )

    #     optimizer_e = torch.optim.AdamW(
    #         euclidean_params,
    #         lr=3e-4,
    #         betas=(0.5, 0.9),
    #         weight_decay=1e-4
    #     )
    # else:
    #     optimizer_e = torch.optim.AdamW(
    #         soundstream.parameters(),
    #         lr=3e-4,
    #         betas=(0.5, 0.9),
    #         weight_decay=1e-4
    #     )
    #     optimizer_g = None
        
    if args.c > 0:
        manifold_params = []
        euclidean_params = []
        for p in soundstream.parameters():
            if hasattr(p, "manifold"):
                manifold_params.append(p)
            else:
                euclidean_params.append(p)
        print(f"Manifold params: {len(manifold_params)}")
        
        param_groups = []
        if len(manifold_params) > 0:
            # Conservative betas for manifold states to reduce exp_avg instability.
            param_groups.append({"params": manifold_params, "lr": args.lr_manifold, "betas": (0.0, 0.95), "eps": args.geoopt_eps})
        if len(euclidean_params) > 0:
            param_groups.append({"params": euclidean_params, "lr": args.lr_g, 'betas': (0.5, 0.9)})

        print(
            f"Geoopt groups: manifold={len(manifold_params)}, "
            f"euclidean={len(euclidean_params)}, lr_manifold={args.lr_manifold:.2e}, lr_g={args.lr_g:.2e}, "
            f"eps={args.geoopt_eps:.1e}, stabilize={args.geoopt_stabilize}, "
            f"quantizer_grad_clip={args.quantizer_grad_clip:.2e}, manifold_grad_clip={args.manifold_grad_clip:.2e}"
        )
        optimizer_g = geoopt.optim.RiemannianAdam(
            param_groups,
            #lr=args.lr_g,
            #betas=(0.5, 0.9),
            #eps=args.geoopt_eps,
            #stabilize=args.geoopt_stabilize,
        )
    else:
        optimizer_g = torch.optim.AdamW(
            soundstream.parameters(), lr=args.lr_g, betas=(0.5, 0.9))
        print('AdamW initialised', args.lr_g)

    # Cosine annealing
    cosine_scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g,
        T_max=args.N_EPOCHS - args.warmup_epochs_g,  # duration AFTER warmup
        eta_min=1e-6  # optional: minimum LR
    )

    exp_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_g, gamma=0.999)

    if args.warmup_epochs_g > 0:
        # Warmup + exponential decay for generator LR
        warmup_scheduler_g = torch.optim.lr_scheduler.LinearLR(
            optimizer_g, start_factor=1e-3, total_iters=args.warmup_epochs_g)
        lr_scheduler_g = torch.optim.lr_scheduler.SequentialLR(
            optimizer_g,
            schedulers=[warmup_scheduler_g, cosine_scheduler_g],
            milestones=[args.warmup_epochs_g])
    else:
        lr_scheduler_g = exp_scheduler_g

    optimizer_d = torch.optim.AdamW(
        itertools.chain(stft_disc.parameters(),
                        msd.parameters(), mpd.parameters()),
        lr=3e-4,
        betas=(0.5, 0.9))
    lr_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_d, gamma=0.999)
    if args.resume:
        latest_info = torch.load(args.resume_path + '/latest.pth')
        args.st_epoch = latest_info['epoch']
        soundstream.load_state_dict(latest_info['soundstream'])
        stft_disc.load_state_dict(latest_info['stft_disc'])
        mpd.load_state_dict(latest_info['mpd'])
        msd.load_state_dict(latest_info['msd'])
        optimizer_g.load_state_dict(latest_info['optimizer_g'])
        lr_scheduler_g.load_state_dict(latest_info['lr_scheduler_g'])
        optimizer_d.load_state_dict(latest_info['optimizer_d'])
        lr_scheduler_d.load_state_dict(latest_info['lr_scheduler_d'])
    train(args, soundstream, stft_disc, msd, mpd, train_loader, valid_loader,
          optimizer_g, optimizer_d, lr_scheduler_g, lr_scheduler_d, logger)


def train(args, soundstream, stft_disc, msd, mpd, train_loader, valid_loader,
          optimizer_g, optimizer_d, lr_scheduler_g, lr_scheduler_d, logger):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from academicodec.visualization import fit_pca, plot_codes, create_gif
    codebook_plots_dir = os.path.join(args.PATH, "codebook_plots")
    print('args ', args.global_rank)
    print('All arguments:')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')

    # Save config to checkpoint directory (rank 0 only)
    if not args.distributed or dist.get_rank() == 0:
        config_path = os.path.join(args.PATH, "config.py")
        with open(config_path, "w") as f:
            f.write("# Training hyperparameters\n")
            for k, v in vars(args).items():
                f.write(f"{k} = {v!r}\n")
        print(f"Config saved to: {config_path}")

    best_val_loss = float("inf")
    best_val_epoch = -1
    global_step = 0
    pca_model = None
    history = {
        "train_rec": [], "train_g": [], "train_d": [], "train_com": [],
        "val_rec": [], "val_g": [], "val_d": [], "val_com": [],
    }
    for epoch in range(args.st_epoch, args.N_EPOCHS + 1):
        soundstream.train()
        stft_disc.train()
        msd.train()
        mpd.train()
        train_loss_d = 0.0
        train_adv_g_loss = 0.0
        train_feat_loss = 0.0
        train_rec_loss = 0.0
        train_loss_g = 0.0
        train_commit_loss = 0.0
        grad_norms_g = []
        grad_norms_d = []
        k_iter = 0
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_pbar = tqdm(train_loader, desc='train')
        for x in train_pbar:
            x = x.to(args.device)
            k_iter += 1
            global_step += 1  # record the global step
            for optimizer_idx in [0, 1]:  # we have two optimizer
                x_wav = get_input(x)
                G_x, commit_loss, last_layer, _ = soundstream(x_wav)
                # --- NaN debug ---
                # import sys
                # if torch.isnan(G_x).any():
                #     print(f"[NaN DEBUG] iter={k_iter} opt={optimizer_idx}: G_x has NaN", file=sys.stderr, flush=True)
                # if torch.isnan(commit_loss).any():
                #     print(f"[NaN DEBUG] iter={k_iter} opt={optimizer_idx}: commit_loss={commit_loss.item()} has NaN", file=sys.stderr, flush=True)
                # if not torch.isnan(G_x).any() and not torch.isnan(commit_loss).any():
                #     if k_iter <= 10:
                #         print(f"[NaN DEBUG] iter={k_iter} opt={optimizer_idx}: G_x OK (max={G_x.abs().max().item():.4f}), commit_loss={commit_loss.item():.4f}", file=sys.stderr, flush=True)
                # --- end NaN debug ---
                if optimizer_idx == 0:
                    # Codebook visualization: refit PCA each time for accurate projection
                    if global_step % args.number_of_steps == 0:
                        if not args.distributed or dist.get_rank() == 0:
                            codebook_module = soundstream.module.quantizer.vq.layers[args.codebook_number]._codebook if args.distributed else soundstream.quantizer.vq.layers[args.codebook_number]._codebook
                            if hasattr(codebook_module, "inited") and codebook_module.inited:
                                codes = codebook_module.embed.detach().cpu()
                                pca_model = fit_pca(codes, args.c)
                                os.makedirs(codebook_plots_dir, exist_ok=True)
                                plot_codes(codes, pca_model, args.c, global_step, codebook_plots_dir)

                    # update generator
                    y_disc_r, fmap_r = stft_disc(x_wav.contiguous())
                    y_disc_gen, fmap_gen = stft_disc(G_x.contiguous())
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                        x_wav.contiguous(), G_x.contiguous())
                    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                        x_wav.contiguous(), G_x.contiguous())
                    total_loss_g, rec_loss, adv_g_loss, feat_loss, d_weight = loss_g(
                        commit_loss,
                        x_wav,
                        G_x,
                        fmap_r,
                        fmap_gen,
                        y_disc_r,
                        y_disc_gen,
                        global_step,
                        y_df_hat_r,
                        y_df_hat_g,
                        y_ds_hat_r,
                        y_ds_hat_g,
                        fmap_f_r,
                        fmap_f_g,
                        fmap_s_r,
                        fmap_s_g,
                        last_layer=last_layer,
                        is_training=True,
                        args=args)
                    train_commit_loss += commit_loss
                    train_loss_g += total_loss_g.item()
                    train_adv_g_loss += adv_g_loss.item()
                    train_feat_loss += feat_loss.item()
                    train_rec_loss += rec_loss.item()

                    # Debug check: catch non-finite quantizer/codebook params before this update.
                    # check_quantizer_finite(soundstream, f"iter={k_iter}/gen/pre_backward")
                    optimizer_g.zero_grad()
                    total_loss_g.backward()

                    # Track generator gradient norm
                    _g_grads = [p.grad.detach().flatten() for p in soundstream.parameters() if p.grad is not None]
                    if _g_grads:
                        grad_norms_g.append(torch.cat(_g_grads).norm().item())

                    # Debug check: detect non-finite gradients produced by backward.
                    # check_quantizer_finite(soundstream, f"iter={k_iter}/gen/post_backward")

                    # Targeted clipping to prevent manifold optimizer state (exp_avg) from blowing up.
                    # q_grad_norm = clip_named_grad_norm(
                    #     soundstream,
                    #     max_norm=args.quantizer_grad_clip,
                    #     include_quantizer=True,
                    #     manifold_only=False,
                    # )
                    # m_grad_norm = clip_named_grad_norm(
                    #     soundstream,
                    #     max_norm=args.manifold_grad_clip,
                    #     include_quantizer=True,
                    #     manifold_only=True,
                    # )
                    # if (k_iter <= 20) and (not args.distributed or dist.get_rank() == 0):
                    #     print(
                    #         f"grad_clip iter={k_iter}: quantizer_norm={q_grad_norm:.4e}, "
                    #         f"manifold_quantizer_norm={m_grad_norm:.4e}",
                    #         flush=True,
                    #     )

                    # torch.nn.utils.clip_grad_norm_(soundstream.parameters(), max_norm=1.0)
                    optimizer_g.step()

                    # Debug check: detect non-finite optimizer state for quantizer/codebook tensors.
                    # check_quantizer_optimizer_state_finite(
                    #     soundstream,
                    #     optimizer_g,
                    #     f"iter={k_iter}/gen/post_step_optimizer_state",
                    # )

                    # Debug check: detect first non-finite values introduced by optimizer step.
                    # check_quantizer_finite(soundstream, f"iter={k_iter}/gen/post_step")
                else:
                    # update discriminator
                    y_disc_r_det, fmap_r_det = stft_disc(x.detach())
                    y_disc_gen_det, fmap_gen_det = stft_disc(G_x.detach())

                    # MPD
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                        x.detach(), G_x.detach())
                    #MSD
                    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                        x.detach(), G_x.detach())

                    loss_d = loss_dis(
                        y_disc_r_det, y_disc_gen_det, fmap_r_det, fmap_gen_det,
                        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r,
                        y_ds_hat_g, fmap_s_r, fmap_s_g, global_step, args)
                    train_loss_d += loss_d.item()
                    optimizer_d.zero_grad()
                    loss_d.backward()

                    # Track discriminator gradient norm
                    _d_grads = [p.grad.detach().flatten()
                                for p in itertools.chain(stft_disc.parameters(), msd.parameters(), mpd.parameters())
                                if p.grad is not None]
                    if _d_grads:
                        grad_norms_d.append(torch.cat(_d_grads).norm().item())

                    # torch.nn.utils.clip_grad_norm_(
                    #     itertools.chain(stft_disc.parameters(), msd.parameters(), mpd.parameters()), max_norm=1.0)
                    optimizer_d.step()
            train_pbar.set_postfix(
                epoch=epoch,
                rec_loss=f'{rec_loss.item():.4f}',
                rec_avg=f'{train_rec_loss / k_iter:.4f}',
                lr_g=f"{optimizer_g.param_groups[0]['lr']:.2e}")
            message = '<epoch:{:d}, iter:{:d}, total_loss_g:{:.4f}, adv_g_loss:{:.4f}, feat_loss:{:.4f}, rec_loss:{:.4f}, commit_loss:{:.4f}, loss_d:{:.4f}, d_weight: {:.4f}>'.format(
                epoch, k_iter,
                total_loss_g.item(),
                adv_g_loss.item(),
                feat_loss.item(),
                rec_loss.item(),
                commit_loss.item(), loss_d.item(), d_weight.item())
            if k_iter % args.print_freq == 0:
                logger.log_info(message)
        lr_scheduler_g.step()
        lr_scheduler_d.step()
        avg_train_rec = train_rec_loss / len(train_loader)
        avg_train_g = train_loss_g / len(train_loader)
        avg_train_d = train_loss_d / len(train_loader)
        avg_train_com = (train_commit_loss / len(train_loader)).item() if torch.is_tensor(train_commit_loss) else train_commit_loss / len(train_loader)
        # Print mean gradient norms for the epoch
        mean_grad_g = sum(grad_norms_g) / len(grad_norms_g) if grad_norms_g else 0.0
        mean_grad_d = sum(grad_norms_d) / len(grad_norms_d) if grad_norms_d else 0.0
        if not args.distributed or dist.get_rank() == 0:
            print(f"[Epoch {epoch}] Mean grad norm  —  generator: {mean_grad_g:.4e}  |  discriminator: {mean_grad_d:.4e}", flush=True)
        message = '<epoch:{:d}, <total_loss_g_train:{:.4f}, recon_loss_train:{:.4f}, adversarial_loss_train:{:.4f}, feature_loss_train:{:.4f}, commit_loss_train:{:.4f}>'.format(
            epoch, avg_train_g, avg_train_rec,
            train_adv_g_loss / len(train_loader),
            train_feat_loss / len(train_loader),
            avg_train_com)
        logger.log_info(message)
        with torch.no_grad():
            soundstream.eval()
            stft_disc.eval()
            mpd.eval()
            msd.eval()
            valid_loss_d = 0.0
            valid_loss_g = 0.0
            valid_commit_loss = 0.0
            valid_adv_g_loss = 0.0
            valid_feat_loss = 0.0
            valid_rec_loss = 0.0
            
            # For tracking perplexity / usage of codebooks
            codes_hist = None
            total_valid_codes = 0
            
            if args.distributed:
                valid_loader.sampler.set_epoch(epoch)
            for x in tqdm(valid_loader):
                x = x.to(args.device)
                for optimizer_idx in [0, 1]:
                    x_wav = get_input(x)
                    # G_x is the reconstructed waveform, codes is the indices
                    G_x, commit_loss, last_layer, codes = soundstream(x_wav)
                    if optimizer_idx == 0:
                        # Tracking Codebook Perplexity
                        # codes shape: [num_quantizers, batch, time]
                        if codes_hist is None:
                            quantizer_module = soundstream.module.quantizer if hasattr(soundstream, 'module') else soundstream.quantizer
                            num_q = quantizer_module.n_q
                            codebook_size = quantizer_module.bins # e.g. 1024
                            codes_hist = torch.zeros(num_q, codebook_size, device=args.device, dtype=torch.float64)
                        
                        # Flatten across batch and time for each quantizer separately
                        for q_idx in range(codes.shape[0]):
                            codes_q = codes[q_idx, :, :].flatten()
                            codes_hist[q_idx].scatter_add_(0, codes_q, torch.ones_like(codes_q, dtype=torch.float64))
                        total_valid_codes += codes.shape[1] * codes.shape[2]

                        valid_commit_loss += commit_loss
                        y_disc_r, fmap_r = stft_disc(x_wav.contiguous())
                        y_disc_gen, fmap_gen = stft_disc(G_x.contiguous())
                        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                            x_wav.contiguous(), G_x.contiguous())
                        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                            x_wav.contiguous(), G_x.contiguous())

                        total_loss_g, adv_g_loss, feat_loss, rec_loss = criterion_g(
                            commit_loss,
                            x_wav,
                            G_x,
                            fmap_r,
                            fmap_gen,
                            y_disc_r,
                            y_disc_gen,
                            y_df_hat_r,
                            y_df_hat_g,
                            fmap_f_r,
                            fmap_f_g,
                            y_ds_hat_r,
                            y_ds_hat_g,
                            fmap_s_r,
                            fmap_s_g,
                            args=args)
                        valid_loss_g += total_loss_g.item()
                        valid_adv_g_loss += adv_g_loss.item()
                        valid_feat_loss += feat_loss.item()
                        valid_rec_loss += rec_loss.item()
                    else:
                        y_disc_r_det, fmap_r_det = stft_disc(
                            x_wav.contiguous().detach())
                        y_disc_gen_det, fmap_gen_det = stft_disc(
                            G_x.contiguous().detach())
                        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                            x_wav.contiguous().detach(),
                            G_x.contiguous().detach())
                        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                            x_wav.contiguous().detach(),
                            G_x.contiguous().detach())
                        loss_d = criterion_d(y_disc_r_det, y_disc_gen_det,
                                             fmap_r_det, fmap_gen_det,
                                             y_df_hat_r, y_df_hat_g, fmap_f_r,
                                             fmap_f_g, y_ds_hat_r, y_ds_hat_g,
                                             fmap_s_r, fmap_s_g)
                        valid_loss_d += loss_d.item()
                        
            # Calculate perplexity: 2^{H(p)}
            perplexities = []
            if codes_hist is not None and total_valid_codes > 0:
                # Distribute stats across nodes if distributed
                if args.distributed:
                    dist.all_reduce(codes_hist, op=dist.ReduceOp.SUM)
                    total_valid_codes_tensor = torch.tensor([total_valid_codes], device=args.device, dtype=torch.float64)
                    dist.all_reduce(total_valid_codes_tensor, op=dist.ReduceOp.SUM)
                    total_valid_codes = total_valid_codes_tensor.item()
                    
                probs = codes_hist / codes_hist.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=-1)
                perplexities = torch.exp2(entropy).tolist()

            avg_val_rec = valid_rec_loss / len(valid_loader)
            avg_val_g = valid_loss_g / len(valid_loader)
            avg_val_d = valid_loss_d / len(valid_loader)
            avg_val_com = (valid_commit_loss / len(valid_loader)).item() if torch.is_tensor(valid_commit_loss) else valid_commit_loss / len(valid_loader)

            # Accumulate loss history
            history["train_rec"].append(avg_train_rec)
            history["train_g"].append(avg_train_g)
            history["train_d"].append(avg_train_d)
            history["train_com"].append(avg_train_com)
            history["val_rec"].append(avg_val_rec)
            history["val_g"].append(avg_val_g)
            history["val_d"].append(avg_val_d)
            history["val_com"].append(avg_val_com)

            # Only save checkpoints after reaching 75% of total epochs
            threshold_epoch = int(0.75 * args.N_EPOCHS)
            if not args.distributed or dist.get_rank() == 0:
                if epoch >= threshold_epoch:
                    latest_save = {
                        'soundstream': soundstream.state_dict(),
                        'stft_disc': stft_disc.state_dict(),
                        'mpd': mpd.state_dict(),
                        'msd': msd.state_dict(),
                        'epoch': epoch,
                        'optimizer_g': optimizer_g.state_dict(),
                        'optimizer_d': optimizer_d.state_dict(),
                        'lr_scheduler_g': lr_scheduler_g.state_dict(),
                        'lr_scheduler_d': lr_scheduler_d.state_dict(),
                    }
                    torch.save(latest_save, os.path.join(args.PATH, 'latest.pth'))

                    if avg_val_rec < best_val_loss:
                        best_val_loss = avg_val_rec
                        best_val_epoch = epoch
                        torch.save(latest_save, os.path.join(args.PATH, f'best_{epoch}.pth'))
                        print(f"  ✓ New best model saved (val_rec={avg_val_rec:.5f})")
                else:
                    # Still track best for logging even before threshold
                    if avg_val_rec < best_val_loss:
                        best_val_loss = avg_val_rec
                        best_val_epoch = epoch

            ppl_str = ", ".join([f"{p:.1f}" for p in perplexities]) if perplexities else "N/A"

            message = '<epoch:{:d}, total_loss_g_valid:{:.4f}, recon_loss_valid:{:.4f}, adversarial_loss_valid:{:.4f}, feature_loss_valid:{:.4f}, commit_loss_valid:{:.4f}, valid_loss_d:{:.4f}, ppl:[{}], best_epoch:{:d}>'.format(
                epoch, avg_val_g, avg_val_rec,
                valid_adv_g_loss / len(valid_loader),
                valid_feat_loss / len(valid_loader),
                avg_val_com,
                avg_val_d, ppl_str, best_val_epoch)
            logger.log_info(message)

    # ── End-of-training: codebook GIF + loss curves (rank 0 only) ──
    if not args.distributed or dist.get_rank() == 0:
        # Final codebook snapshot + GIF
        if pca_model is not None:
            codebook_module = soundstream.module.quantizer.vq.layers[args.codebook_number]._codebook if args.distributed else soundstream.quantizer.vq.layers[args.codebook_number]._codebook
            codes = codebook_module.embed.detach().cpu()
            plot_codes(codes, pca_model, args.c, global_step, codebook_plots_dir)
        gif_path = os.path.join(args.PATH, "codebook_evolution.gif")
        create_gif(codebook_plots_dir, gif_path)

        # ── Loss curves ──────────────────────────────────────────────
        if len(history["train_rec"]) > 0:
            epochs_range = list(range(args.st_epoch, args.st_epoch + len(history["train_rec"])))

            fig, axes = plt.subplots(2, 2, figsize=(16, 10))

            axes[0, 0].plot(epochs_range, history["train_rec"], label="Train")
            axes[0, 0].plot(epochs_range, history["val_rec"], label="Val")
            axes[0, 0].set_title("Reconstruction Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            axes[0, 1].plot(epochs_range, history["train_com"], label="Train")
            axes[0, 1].plot(epochs_range, history["val_com"], label="Val")
            axes[0, 1].set_title("Commitment Loss")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            axes[1, 0].plot(epochs_range, history["train_g"], label="Train")
            axes[1, 0].plot(epochs_range, history["val_g"], label="Val")
            axes[1, 0].set_title("Generator Loss")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            axes[1, 1].plot(epochs_range, history["train_d"], label="Train")
            axes[1, 1].plot(epochs_range, history["val_d"], label="Val")
            axes[1, 1].set_title("Discriminator Loss")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

            fig.tight_layout()
            loss_fig_path = os.path.join(args.PATH, "loss_curves.png")
            fig.savefig(loss_fig_path, dpi=150)
            plt.close(fig)
            print(f"Loss curves saved to: {loss_fig_path}")



if __name__ == '__main__':
    main()
