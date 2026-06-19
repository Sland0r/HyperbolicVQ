# 与 Encodec_24k_240d main3_ddp.py 相比只有鉴别器不同
import argparse
import math as _math
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
        p = param.tensor if hasattr(param, 'tensor') else param
        param_size += p.nelement() * p.element_size()
        param_sum += p.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        b = buffer.tensor if hasattr(buffer, 'tensor') else buffer
        buffer_size += b.nelement() * b.element_size()
        buffer_sum += b.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

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
    parser.add_argument('--bins', type=int, default=1024, help='number of bins')
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
        '--decay',
        type=float,
        default=0.99,
        help='decay rate for EMA (default: 0.99)')
    parser.add_argument(
        '--exponential_lambda',
        type=float,
        default=0.4,
        help='exponential_lambda of codebook dropout')
    parser.add_argument(
        '--codebook_weight',
        type=float,
        default=1.0,
        help='weight of codebook loss')
    parser.add_argument(
        '--commitment_weight',
        type=float,
        default=0.25,
        help='weight of commitment loss')
    parser.add_argument(
        '--remove',
        type=int,
        default=0,
        help='number of codebooks to remove (default: 0)')
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
    parser.add_argument(
        '--use_spec_augment',
        action='store_true',
        help='apply SpecAugment data augmentation (default: False)')
    parser.add_argument(
        '--dot_product_weight',
        type=float,
        default=0.0,
        help='dot_product_weight for codebook (default: 0.0)')
    parser.add_argument(
        '--entailment_cone_weight',
        type=float,
        default=0.0,
        help='entailment_cone_weight for codebook (default: 0.0)')
    parser.add_argument(
        '--gyration_weight',
        type=float,
        default=0.0,
        help='gyration penalty weight: penalizes sin(angle) between code and residual to minimize gyration rotation (default: 0.0)')
    parser.add_argument(
        '--LAMBDA_SEP',
        type=float,
        default=0.0,
        help='hyper-parameter for codebook separation loss (0 = disabled)')
    parser.add_argument(
        '--uniform',
        action='store_true',
        help='uniform codebook dropout')
    parser.add_argument(
        '--constructive',
        action='store_true',
        help='initialize codebooks using constructive tree embeddings (depth=1)')
    parser.add_argument(
        '--codebook_dim',
        type=int,
        default=512,
        help='codebook dimension (default: None, same as bottleneck D=512). '
             'If lower than D, a linear projection is added. '
             'When c>0, a hyperbolic (HypLinear) projection is used.')
    parser.add_argument(
        '--threshold_ema_dead_code',
        type=int,
        default=2,
        help='threshold for EMA dead code replacement (default: 2)')
    parser.add_argument(
        '--solution',
        action='store_true',
        help='Use Solution 1: Tangent Space Residuals via Parallel Transport')
    parser.add_argument(
        '--gyration',
        action='store_true',
        help='Use explicit gyrovector algebra for gyration correction')
    parser.add_argument(
        '--parallel_transport',
        action='store_true',
        help='Use pure differential geometry approach with parallel transport')
    parser.add_argument(
        '--new_method', action='store_true',
        help='Use left-subtraction encoding with right-associative decoding (default: False)')
    parser.add_argument(
        '--approx', action='store_true',
        help='Track hyperbolic approximation distance (quantized+residual vs input)')
    parser.add_argument(
        '--hste', action='store_true',
        help='Use hyperbolic straight-through estimator instead of Euclidean STE')
    parser.add_argument(
        '--hste_riemannian', action='store_true',
        help='Geometry-exact discount: HyperbolicSTE.backward returns the Riemannian gradient at x. Requires --hste.')
    parser.add_argument(
        '--gradient_correction', action='store_true',
        help='Detach the residual after each quantization step (gradient correction)')
    parser.add_argument(
        '--A5', action='store_true',
        help='Commitment loss only at the first and last RQ steps; the last one '
             'is computed on a never-detached residual chain so its gradient '
             'reaches the encoder through all Mobius updates even under '
             '--gradient_correction (combine with the A4 config).')
    parser.add_argument(
        '--A4_v2', action='store_true',
        help='Strict gradient correction: also detach the FIRST residual so NO '
             'per-layer commitment loss reaches the encoder (vanilla gc still '
             'leaks layer 0 through r0). Requires the A4 config; with --A5 leaves '
             'only the last-layer commit chain.')
    parser.add_argument(
        '--a6', action='store_true',
        help='Keep-first recon routing: only layer 0 carries the reconstruction '
             'gradient to the encoder (eq:stack i=1 term). Pure recon router; '
             'leaves all commitment losses intact (unlike gc). c>0 + per-layer '
             'STE; inert under a block STE. Mutually exclusive with --a7.')
    parser.add_argument(
        '--a7', action='store_true',
        help='Keep-last recon routing: only the last layer carries the recon '
             'gradient (eq:stack i=N term, full transport chain). Per-layer '
             'realization of the A4 block hop. Run with gc OFF (gc severs the '
             'chain). Mutually exclusive with --a6.')
    parser.add_argument(
        '--a6.1', dest='a6_1', action='store_true',
        help='SUM routing: encoder recon gradient = A4 block-hop grad + a6 '
             '(keep-first) per-layer grad, added. Requires --hste; keeps per-layer '
             'codes attached. Mutually exclusive with a6/a7/a7.1.')
    parser.add_argument(
        '--a7.1', dest='a7_1', action='store_true',
        help='SUM routing: encoder recon gradient = A4 block-hop grad + a7 '
             '(keep-last) per-layer grad, added. Requires --hste. Mutually '
             'exclusive with a6/a7/a6.1.')
    parser.add_argument(
        '--a8', action='store_true',
        help='FULL-SUM routing: A4 block-hop grad + ALL per-layer recon grad '
             '(un-truncated sibling of a6.1/a7.1). Requires --new_method; keeps '
             'per-layer codes attached. Mutually exclusive with a6/a7/a6.1/a7.1.')
    parser.add_argument(
        '--embed_init_scale', type=float, default=1.0,
        help='Scale factor for embedding initialization')
    parser.add_argument(
        '--gyration_only', action='store_true',
        help='(with --hste) HSTE backward = pure gyration transport of the gradient '
             'from q to x: geometric direction correction, NO conformal lambda '
             '("gamma") coefficients anywhere, Euclidean magnitude preserved exactly. '
             'Takes precedence over --hste_riemannian.')
    parser.add_argument(
        '--block_hste', action='store_true',
        help='(c>0) One identity STE wrapping the whole RVQ block in tangent space; '
             'per-layer codes are detached. Decoder gradient reaches the encoder '
             'with factor exactly 1. Needs --tangent_proj when codebook_dim != D.')
    parser.add_argument(
        '--block_hste_pt', action='store_true',
        help='(c>0) Like --block_hste, but the single block-level hop is a '
             'HyperbolicSTE transport on the ball from the full Mobius sum Q back '
             'to the initial residual r0 (applied before project_out/log_map0, so '
             'no --tangent_proj restriction). Per-layer codes are detached. The '
             'STE lie d(Q, r0) = d(r_N, 0) equals the last-layer quantization '
             'error; net conformal gain is the endpoint ratio lambda_r0/lambda_Q. '
             '--hste_riemannian/--gyration_only apply to the hop. '
             'Mutually exclusive with --block_hste.')
    parser.add_argument(
        '--hste_clip', action='store_true',
        help='(c>0) HSTE backward: cap the per-vector outgoing grad norm at the '
             'incoming norm (net conformal amplification <= 1, direction '
             'preserved). Alternative to --hste_riemannian on the (block) hop.')
    parser.add_argument(
        '--leaky_clamp', action='store_true',
        help='(c>0) Replace the hard atanh clamp in hyperbolic_distance_sq with '
             'a C1 linear extension beyond 1-1e-3, so saturated commit/codebook '
             'distances keep a restoring gradient instead of going flat '
             '(anti-collapse). Off by default; affects only this run.')
    parser.add_argument(
        '--tangent_proj', action='store_true',
        help='(c>0 only) Put the codebook_dim bottleneck as a Euclidean nn.Linear '
             'BEFORE exp_map0 (and after log_map0), so exp_map/codebooks/quantization '
             'live in the low-dim Poincare ball. No hyperbolic HLinear layers.')
    parser.add_argument(
        '--init_scale', type=float, default=1.0,
        help='Scale factor for constructive initialization')
    parser.add_argument(
        '--compile', action='store_true',
        help='Wrap model in torch.compile to fuse hyperbolic kernels')
    parser.add_argument(
        '--encoder_scale', type=float, default=1.0,
        help='Scale applied to the tangent vector before exp_map0 (c>0 only). '
             '>1 pushes hyperbolic residuals off the origin so the nearest-code '
             'argmax can discriminate by direction (fixes val codebook collapse). '
             '<=0 enables one-time first-batch auto-calibration: a single global '
             'scale is fitted so the median residual lands at radius 0.5 (interior), '
             'then frozen. Keeps per-vector magnitude variation, unlike --encoder_shell.')
    parser.add_argument(
        '--encoder_scale_ema', type=float, default=0.0,
        help='If >0 (c>0 only, requires --encoder_scale<=0), the auto-calibrated '
             'global tangent scale is NOT frozen after the first batch but tracked '
             'with this EMA decay: each training step the scale is nudged so the '
             'median residual stays at radius 0.5 as the encoder output-magnitude '
             'distribution drifts during training. Typical value 0.99.')
    parser.add_argument(
        '--encoder_shell', type=float, default=0.0,
        help='If >0 (c>0 only), L2-normalise each encoder output so exp_map0 lands '
             'on a fixed ball-radius = encoder_shell/sqrt(c). Removes the magnitude '
             'DOF so the encoder cannot collapse residuals to the origin. Takes '
             'precedence over --encoder_scale.')
    parser.add_argument(
        '--code_max_radius', type=float, default=0.0,
        help='If >0 (c>0 only), cap codebook embeddings at this fraction of the '
             'ball radius. Keeps codes off the boundary, where hyperbolic_distance_sq '
             'saturates its atanh clamp and zeroes the commit/codebook gradients '
             '(the cause of hyperbolic codebook collapse).')
    parser.add_argument(
        '--target_max_recon', type=float, default=0.0,
        help='If >0 (c>0 only), set --code_max_radius automatically from n_q so the '
             'worst-case (all codes collinear) Mobius-sum reconstruction lands at '
             'this fraction of the ball radius: cap = tanh(atanh(T)/n_q). '
             'Curvature-independent; overrides any explicit --code_max_radius.')
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
    return x.to(torch.get_default_dtype())


def main():
    args = get_args()
    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)
    if args.num_node == 1:
        args.dist_url = "auto"
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

def main_worker(local_rank, args):
    # if getattr(args, 'c', 0.0) > 0:
    #     torch.set_default_dtype(torch.float64)
    args.local_rank = local_rank
    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1
    # torch.autograd.set_detect_anomaly(True)
    #CUDA_VISIBLE_DEVICES = int(args.local_rank)
    logger = Logger(args)
    if args.leaky_clamp:
        from academicodec.quantization import core_vq as _core_vq
        _core_vq.set_leaky_dist_clamp(True)
        print('[leaky_clamp] C1 atanh extension active in hyperbolic_distance_sq')
    # 240倍下采
    # if args.constructive:
    #     args.threshold_ema_dead_code = 0
    soundstream = SoundStream(n_filters=32, D=512, target_bandwidths=args.target_bandwidths, exponential_lambda=args.exponential_lambda,
                              uniform=args.uniform, threshold_ema_dead_code=args.threshold_ema_dead_code,
                              codebook_weight=args.codebook_weight, commitment_weight=args.commitment_weight,
                              dot_product_weight=args.dot_product_weight, entailment_cone_weight=args.entailment_cone_weight,
                              gyration_weight=args.gyration_weight,
                              ratios=args.ratios, decay=args.decay,
                              sample_rate=args.sr, bins=args.bins, c=args.c, ema=args.ema, kmeans_init=args.kmeans_init,
                              pre_quant_batchnorm=args.pre_quant_batchnorm, remove=args.remove,
                              codebook_dim=args.codebook_dim, new_method=args.new_method, approx=args.approx, hste=args.hste, hste_riemannian=args.hste_riemannian,
                              hste_clip=args.hste_clip,
                              gyration_only=args.gyration_only, block_hste=args.block_hste,
                              block_hste_pt=args.block_hste_pt,
                              gradient_correction=args.gradient_correction, A5=args.A5, A4_v2=args.A4_v2,
                              a6=args.a6, a7=args.a7, a6_1=args.a6_1, a7_1=args.a7_1, a8=args.a8,
                              encoder_scale=args.encoder_scale, encoder_scale_ema=args.encoder_scale_ema,
                              encoder_shell=args.encoder_shell, code_max_radius=args.code_max_radius,
                              target_max_recon=args.target_max_recon,
                              embed_init_scale=args.embed_init_scale, tangent_proj=args.tangent_proj)
    #print(soundstream)
    msd = MultiScaleDiscriminator()
    mpd = MultiPeriodDiscriminator()
    stft_disc = MultiScaleSTFTDiscriminator(filters=32)
    getModelSize(soundstream)
    getModelSize(msd)
    getModelSize(mpd)
    getModelSize(stft_disc)

    # ── Constructive codebook initialisation ──────────────────────
    if args.constructive:
        import sys
        sys.path.insert(0, '/home/acolombo/music/hyperbolic_tree_embeddings')
        from tree_embeddings.trees.file_utils import load_hierarchy
        from tree_embeddings.embeddings.constructive_method import constructively_embed_tree

        _D = 512
        _bins = soundstream.bins
        _n_q = soundstream.n_q

        hierarchy = load_hierarchy(dataset="n_h_trees", hierarchy_name=f"{_bins}_1")

        curvature = args.c if args.c > 0 else 1.0
        try:
            embeddings, _, _ = constructively_embed_tree(
                hierarchy=hierarchy,
                dataset="n_h_trees",
                hierarchy_name=f"{_bins}_1",
                embedding_dim=_D,
                tau=1.0,
                nc=1,
                curvature=curvature,
                root=0,
                gen_type="optim",
                dtype=torch.float32,
            )
        finally:
            # The sphere-point optimiser inside constructively_embed_tree flips on
            # torch.autograd.set_detect_anomaly(True) globally and never resets it,
            # which makes every later training backward pass crawl. Force it off.
            torch.autograd.set_detect_anomaly(False)

        # Skip root (index 0, at origin); keep the bins children
        code_points = embeddings[1:].to(dtype=torch.float32) * (args.init_scale / _n_q)  # (bins, D)
        assert code_points.shape == (_bins, _D), \
            f"Expected ({_bins}, {_D}), got {code_points.shape}"

        # Copy same points into every codebook
        with torch.no_grad():
            for qi in range(_n_q):
                cb = soundstream.quantizer.vq.layers[qi]._codebook
                cb.embed.data.copy_(code_points)
                cb.embed_avg.data.copy_(code_points)
                cb.inited.data.copy_(torch.Tensor([True]))
                cb.cluster_size.data.fill_(args.threshold_ema_dead_code + 1)

        print(
            f"Constructive init: {code_points.shape} points copied to all "
            f"{_n_q} codebooks (curvature={curvature}, tau=1.0)"
        )

    if args.distributed:
        soundstream = torch.nn.SyncBatchNorm.convert_sync_batchnorm(soundstream)
        stft_disc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(stft_disc)
        msd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(msd)
        mpd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(mpd)
    # torch.distributed.barrier()
    if getattr(args, 'compile', False):
        soundstream = torch.compile(soundstream)
        stft_disc = torch.compile(stft_disc)
        msd = torch.compile(msd)
        mpd = torch.compile(mpd)
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

    if args.train_data_path == "100":
        args.train_data_path = "/home/acolombo/VAEs/dataset/LibriTTS/train-clean-100"
    elif args.train_data_path == "360":
        args.train_data_path = "/scratch-shared/acolombo/LibriTTS/train-clean-360"
    elif args.train_data_path == "500":
        args.train_data_path = "/scratch-shared/acolombo/LibriTTS/train-other-500"
    else:
        raise ValueError("Invalid train_data_path")

    train_dataset = NSynthDataset(audio_dir=args.train_data_path, use_spec_augment=args.use_spec_augment)
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
        )
        # Debug: locate codebook embed param and check optimizer inclusion
        def _find_codebook_param(model):
            module = model.module if hasattr(model, 'module') else model
            try:
                codebook_module = module.quantizer.vq.layers[args.codebook_number]._codebook
                embed_param = getattr(codebook_module, 'embed', None)
                return codebook_module, embed_param
            except Exception:
                return None, None

        cb_module, cb_param = _find_codebook_param(soundstream)
        if cb_param is None:
            print('DEBUG: codebook embed param not found', flush=True)
        else:
            found = False
            if hasattr(optimizer_g, 'param_groups'):
                for i, g in enumerate(optimizer_g.param_groups):
                    params = g.get('params', [])
                    if any(p is cb_param for p in params):
                        print(f'DEBUG: codebook embed param found in optimizer group {i} (lr={g.get("lr")})', flush=True)
                        found = True
            if not found:
                print('DEBUG: codebook embed param NOT in any optimizer param group', flush=True)
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
    print('Len dataloader ', len(train_loader))
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
        "train_rec": [], "train_g": [], "train_d": [], "train_com": [], "train_feat": [],
        "val_rec": [], "val_g": [], "val_d": [], "val_com": [], "val_feat": [],
        "train_dist": [], "val_dist": [],
    }
    epochs = range(args.st_epoch, args.N_EPOCHS + 1)
    for epoch in tqdm(epochs, desc='epoch'):
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
        train_sep_sum = 0.0
        train_dist_sum = 0.0
        grad_norms_g = []
        grad_norms_d = []
        k_iter = 0
        total_train_iters = len(train_loader)
        last_10_percent_start = int(total_train_iters * 0.9)
        train_codes_hist = None
        train_codes_hist_last10 = None
        total_train_codes = 0
        total_train_codes_last10 = 0
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        #train_pbar = tqdm(train_loader, desc='train')
        #for x in train_pbar:
        for x in train_loader:
            x = x.to(args.device)
            k_iter += 1
            global_step += 1  # record the global step
            for optimizer_idx in [0, 1]:  # we have two optimizer
                x_wav = get_input(x)
                G_x, commit_loss, last_layer, codes, distance = soundstream(x_wav)
                if optimizer_idx == 0:
                    with torch.no_grad():
                        if train_codes_hist is None:
                            quantizer_module = soundstream.module.quantizer if hasattr(soundstream, 'module') else soundstream.quantizer
                            num_q = quantizer_module.n_q
                            codebook_size = quantizer_module.bins
                            train_codes_hist = torch.zeros(num_q, codebook_size, device=args.device, dtype=torch.float64)
                            train_codes_hist_last10 = torch.zeros(num_q, codebook_size, device=args.device, dtype=torch.float64)
                        
                        codes_count = codes.shape[1] * codes.shape[2]
                        total_train_codes += codes_count
                        is_last10 = k_iter >= last_10_percent_start
                        if is_last10:
                            total_train_codes_last10 += codes_count
                        
                        for q_idx in range(codes.shape[0]):
                            codes_q = codes[q_idx, :, :].flatten()
                            ones = torch.ones_like(codes_q, dtype=torch.float64)
                            train_codes_hist[q_idx].scatter_add_(0, codes_q, ones)
                            if is_last10:
                                train_codes_hist_last10[q_idx].scatter_add_(0, codes_q, ones)

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
                    train_commit_loss += commit_loss.item()
                    train_loss_g += total_loss_g.item()
                    train_adv_g_loss += adv_g_loss.item()
                    train_feat_loss += feat_loss.item()
                    train_rec_loss += rec_loss.item()
                    train_dist_sum += distance.item()

                    # ── Codebook separation loss ──────────────────────────
                    sep_loss_val = 0.0
                    if args.LAMBDA_SEP > 0 and not args.ema:
                        from academicodec.quantization.core_vq import pairwise_hyperbolic_distance_sq
                        sep_loss = torch.tensor(0.0, device=args.device)
                        ss_module = soundstream.module if hasattr(soundstream, 'module') else soundstream
                        n_q_active = len(ss_module.quantizer.vq.layers)
                        for qi in range(n_q_active):
                            embed = ss_module.quantizer.vq.layers[qi]._codebook.embed  # (codebook_size, D)
                            if args.c > 0:
                                dist_matrix = pairwise_hyperbolic_distance_sq(embed, embed, args.c)
                            else:
                                diff = embed.unsqueeze(0) - embed.unsqueeze(1)  # (K, K, D)
                                dist_matrix = diff.pow(2).sum(-1)               # (K, K)
                            sep_loss = sep_loss - dist_matrix.sum()
                        sep_loss = args.LAMBDA_SEP * sep_loss
                        sep_loss_val = sep_loss.item()
                        total_loss_g = total_loss_g + sep_loss
                    train_sep_sum += sep_loss_val

                    # Debug check: catch non-finite quantizer/codebook params before this update.
                    # check_quantizer_finite(soundstream, f"iter={k_iter}/gen/pre_backward")
                    optimizer_g.zero_grad()
                    total_loss_g.backward()

                    # Debug: print codebook embed gradient info (if available)
                    try:
                        cb_param  # ensure name is in scope
                    except NameError:
                        cb_param = None

                    if cb_param is not None:
                        grad = getattr(cb_param, 'grad', None)
                        if grad is None:
                            print(f'DEBUG: codebook embed grad is None at iter {global_step}', flush=True)
                        else:
                            try:
                                gn = grad.detach().norm().item()
                            except Exception:
                                gn = float('nan')
                            print(f'DEBUG: codebook embed grad norm {gn:.6g} at iter {global_step}', flush=True)

                    # Track generator gradient norm
                    _g_grads = [p.grad.detach().flatten() for p in soundstream.parameters() if p.grad is not None]
                    if _g_grads:
                        grad_norms_g.append(torch.cat(_g_grads).norm().item())
                    optimizer_g.step()
                else:
                    # update discriminator
                    y_disc_r_det, fmap_r_det = stft_disc(x_wav.contiguous().detach())
                    y_disc_gen_det, fmap_gen_det = stft_disc(G_x.contiguous().detach())

                    # MPD
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                        x_wav.contiguous().detach(), G_x.contiguous().detach())
                    #MSD
                    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                        x_wav.contiguous().detach(), G_x.contiguous().detach())

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

                    optimizer_d.step()
            # train_pbar.set_postfix(
            #     epoch=epoch,
            #     rec_loss=f'{rec_loss.item():.4f}',
            #     rec_avg=f'{train_rec_loss / k_iter:.4f}',
            #     lr_g=f"{optimizer_g.param_groups[0]['lr']:.2e}")
            message = '<epoch:{:d}, iter:{:d}, total_loss_g:{:.4f}, adv_g_loss:{:.4f}, feat_loss:{:.4f}, rec_loss:{:.4f}, commit_loss:{:.4f}, loss_d:{:.4f}, d_weight: {:.4f}, sep_loss:{:.6f}>'.format(
                epoch, k_iter,
                total_loss_g.item(),
                adv_g_loss.item(),
                feat_loss.item(),
                rec_loss.item(),
                commit_loss.item(), loss_d.item(), d_weight.item(), sep_loss_val)
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
            
        with torch.no_grad():
            train_ppl_whole = []
            if train_codes_hist is not None and total_train_codes > 0:
                if args.distributed:
                    dist.all_reduce(train_codes_hist, op=dist.ReduceOp.SUM)
                probs = train_codes_hist / train_codes_hist.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=-1)
                train_ppl_whole = torch.exp2(entropy).tolist()

            train_ppl_last10 = []
            if train_codes_hist_last10 is not None and total_train_codes_last10 > 0:
                if args.distributed:
                    dist.all_reduce(train_codes_hist_last10, op=dist.ReduceOp.SUM)
                probs = train_codes_hist_last10 / train_codes_hist_last10.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=-1)
                train_ppl_last10 = torch.exp2(entropy).tolist()

        avg_train_sep = train_sep_sum / len(train_loader)
        message = '<epoch:{:d}, <total_loss_g_train:{:.4f}, recon_loss_train:{:.4f}, adversarial_loss_train:{:.4f}, feature_loss_train:{:.4f}, commit_loss_train:{:.4f}, sep_loss_train:{:.6f}>'.format(
            epoch, avg_train_g, avg_train_rec,
            train_adv_g_loss / len(train_loader),
            train_feat_loss / len(train_loader),
            avg_train_com, avg_train_sep)
        logger.log_info(message)
        train_ppl_str = ", ".join([f"{p:.1f}" for p in train_ppl_whole]) if train_ppl_whole else "N/A"
        train_ppl_last10_str = ", ".join([f"{p:.1f}" for p in train_ppl_last10]) if train_ppl_last10 else "N/A"
        logger.log_info(f"Train PPL (whole epoch): [{train_ppl_str}]")
        logger.log_info(f"Train PPL (last 10%): [{train_ppl_last10_str}]")

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
            valid_dist_sum = 0.0
            valid_dist_n = 0
            
            if args.distributed:
                valid_loader.sampler.set_epoch(epoch)
            #for x in tqdm(valid_loader):
            for x in valid_loader:
                x = x.to(args.device)
                for optimizer_idx in [0, 1]:
                    x_wav = get_input(x)
                    # G_x is the reconstructed waveform, codes is the indices
                    G_x, commit_loss, last_layer, codes, distance = soundstream(x_wav, validation=True)
                    if optimizer_idx == 0:
                        valid_dist_sum += distance.item()
                        valid_dist_n += 1
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
                        
            avg_val_dist = valid_dist_sum / max(valid_dist_n, 1)
            avg_train_dist = train_dist_sum / len(train_loader)
            if args.approx:
                logger.log_info(f"  [Epoch {epoch}] Approx distance — train: {avg_train_dist:.6f}  val: {avg_val_dist:.6f}")

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
            avg_train_feat = train_feat_loss / len(train_loader)
            avg_val_feat = valid_feat_loss / len(valid_loader)

            # Accumulate loss history
            history["train_rec"].append(avg_train_rec)
            history["train_g"].append(avg_train_g)
            history["train_d"].append(avg_train_d)
            history["train_com"].append(avg_train_com)
            history["train_feat"].append(avg_train_feat)
            history["val_rec"].append(avg_val_rec)
            history["val_g"].append(avg_val_g)
            history["val_d"].append(avg_val_d)
            history["val_com"].append(avg_val_com)
            history["val_feat"].append(avg_val_feat)
            history["train_dist"].append(avg_train_dist)
            history["val_dist"].append(avg_val_dist)

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

    # ── End-of-training summary ──
    if not args.distributed or dist.get_rank() == 0:
        logger.log_info(f"\n  Last model  (epoch {epoch}):  val_rec={avg_val_rec:.5f}  "
              f"val_com={avg_val_com:.5f}  val_approx_dist={avg_val_dist:.5f}")

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

            axes[1, 0].plot(epochs_range, history["train_feat"], label="Train")
            axes[1, 0].plot(epochs_range, history["val_feat"], label="Val")
            axes[1, 0].set_title("Feature Loss")
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

            if args.approx and any(d != 0.0 for d in history["train_dist"]):
                fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
                ax_dist.plot(epochs_range, history["train_dist"], label="Train")
                ax_dist.plot(epochs_range, history["val_dist"], label="Val")
                ax_dist.set_title("Hyperbolic Approximation Distance")
                ax_dist.set_xlabel("Epoch")
                ax_dist.set_ylabel("Mean Hyperbolic Distance²")
                ax_dist.legend()
                ax_dist.grid(True)
                fig_dist.tight_layout()
                dist_fig_path = os.path.join(args.PATH, "approx_distance.png")
                fig_dist.savefig(dist_fig_path, dpi=150)
                plt.close(fig_dist)
                print(f"Approx distance curve saved to: {dist_fig_path}")


if __name__ == '__main__':
    main()
