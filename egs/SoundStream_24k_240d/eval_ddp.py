import argparse
import itertools
import os
import time

import torch
import torch.distributed as dist
import math
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
    parser.add_argument('--BATCH_SIZE', type=int, default=2, help='batch size')
    parser.add_argument(
        '--PATH',
        type=str,
        default='model_path/',
        help='The path to save the model')
    parser.add_argument('--sr', type=int, default=24000, help='sample rate')
    parser.add_argument(
        '--print_freq', type=int, default=10, help='the print number')
    parser.add_argument(
        '--save_dir', type=str, default='log', help='log save path')
    parser.add_argument(
        '--valid_data_path',
        type=str,
        default='path_to_val_wavs',
        help='validation data')
    parser.add_argument(
        '--checkpoint', type=str, default='path_to_resume', help='checkpoint_path to evaluate')
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
    args = parser.parse_args()
    
    if 'SLURM_JOB_ID' in os.environ:
        time_str = os.environ['SLURM_JOB_ID'] + "_eval"
    else:
        time_str = time.strftime('%Y-%m-%d-%H-%M') + "_eval"
        
    args.save_dir = os.path.join(args.save_dir, time_str)
    os.makedirs(args.save_dir, exist_ok=True)
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


def main_worker(local_rank, args):
    args.local_rank = local_rank
    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1
    logger = Logger(args)
    # 240倍下采
    soundstream = SoundStream(n_filters=32, D=512, ratios=args.ratios, c=args.c, ema=args.ema)
    msd = MultiScaleDiscriminator()
    mpd = MultiPeriodDiscriminator()
    stft_disc = MultiScaleSTFTDiscriminator(filters=32)

    if args.distributed:
        soundstream = torch.nn.SyncBatchNorm.convert_sync_batchnorm(soundstream)
        stft_disc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(stft_disc)
        msd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(msd)
        mpd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(mpd)
        
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

    valid_dataset = NSynthDataset(audio_dir=args.valid_data_path)
    args.sr = valid_dataset.sr
    if args.distributed:
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset)
    else:
        valid_sampler = None
        
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=8,
        sampler=valid_sampler)
        
    logger.log_info(f"Loading checkpoint from: {args.checkpoint}")
    latest_info = torch.load(args.checkpoint, map_location=args.device)
    
    # Handle the fact that we might be loading only generator, or both 
    if 'soundstream' in latest_info:
        soundstream.load_state_dict(latest_info['soundstream'])
        if 'stft_disc' in latest_info:
            stft_disc.load_state_dict(latest_info['stft_disc'])
        if 'mpd' in latest_info:
            mpd.load_state_dict(latest_info['mpd'])
        if 'msd' in latest_info:
            msd.load_state_dict(latest_info['msd'])
    else:
        # assume it's just the generator weights directly
        soundstream.load_state_dict(latest_info)

    evaluate(args, soundstream, stft_disc, msd, mpd, valid_loader, logger)


def evaluate(args, soundstream, stft_disc, msd, mpd, valid_loader, logger):
    logger.log_info(f"Starting evaluation on {len(valid_loader)} batches")
    print('All arguments:')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')
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
        
        for x in tqdm(valid_loader):
            x = x.to(args.device)
            # We do both optimizer_idx steps just to compute all the metrics the same way
            for optimizer_idx in [0, 1]:
                x_wav = get_input(x)
                # G_x is the reconstructed waveform, codes is the indices
                if optimizer_idx == 0:
                    G_x, commit_loss, last_layer, codes = soundstream(x_wav)
                    # Tracking Codebook Perplexity
                    # codes shape: [batch, num_quantizers, time]
                    if codes_hist is None:
                        # Use absolute maximum n_q from the module, NOT the batch shape which is truncated by bandwidth dropout
                        quantizer_module = soundstream.module.quantizer if hasattr(soundstream, 'module') else soundstream.quantizer
                        num_q = quantizer_module.n_q 
                        codebook_size = quantizer_module.bins
                        codes_hist = torch.zeros(num_q, codebook_size, device=args.device, dtype=torch.float64)
                    
                    # Flatten across batch and time for each quantizer separately
                    for q_idx in range(codes.shape[0]):
                        codes_q = codes[q_idx, :, :].flatten()
                        codes_hist[q_idx].scatter_add_(0, codes_q, torch.ones_like(codes_q, dtype=torch.float64))
                    total_valid_codes += codes.shape[1] * codes.shape[2]

                    valid_commit_loss += commit_loss.item()
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
                    G_x, _, _, _ = soundstream(x_wav)
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

        ppl_str = ", ".join([f"{p:.1f}" for p in perplexities]) if perplexities else "N/A"

        message = '<EVALUATION RESULTS>: total_loss_g_valid:{:.4f}, recon_loss_valid:{:.4f}, adversarial_loss_valid:{:.4f}, feature_loss_valid:{:.4f}, commit_loss_valid:{:.4f}, valid_loss_d:{:.4f}, ppl:[{}]>'.format(
            valid_loss_g / len(valid_loader), valid_rec_loss /
            len(valid_loader), valid_adv_g_loss / len(valid_loader),
            valid_feat_loss / len(valid_loader),
            valid_commit_loss / len(valid_loader),
            valid_loss_d / len(valid_loader), ppl_str)
        logger.log_info(message)
        print("\n\n" + "="*80)
        print(message)
        print("="*80 + "\n\n")

if __name__ == '__main__':
    main()
