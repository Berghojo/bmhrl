import numpy as np
import torch
from torch.utils import tensorboard as tensorboard

# Provided by Iashin and Rahtu at https://github.com/v-iashin/BMT
from torch.utils.data import DataLoader
from utilities.learning import adjust_optimizer_lr
from epoch_loops.validation_loops import validation_1by1_loop
from captioning_datasets.captioning_dataset import ActivityNetCaptionsDataset
from loss.label_smoothing import LabelSmoothing
from utilities.captioning_utils import average_metrics_in_two_dicts, timer
#----------------------------------------------------------------

from epoch_loops.captioning_bmrl_loops import bimodal_decoder, audio_decoder, video_decoder,\
    bmhrl_validation_next_word_loop, train_bmhrl_bl, warmstart_bmhrl_bl, train_audio_bl, train_video_bl, \
    warmstart_audio_bl, warmstart_video_bl, analyze_bmhrl_div, train_detr_rl, reinforce_detr_rl, detr_decoder
from metrics.batched_meteor import MeteorScorer
from metrics.cider import CiderScorer
from metrics.bleu import BleuScorer
from model.bm_hrl_agent import BMHrlAgent, BMManagerValueFunction, BMWorkerValueFunction, AudioAgent, VideoAgent
from model.det_bmhrl_agent import DetrCaption
from utilities.out_log import print_to_file as print_log
from epoch_loops.captioning_rl_loops import (rl_training_loop, inference, validation_next_word_loop, warmstart, rl_likelyhood)
from loss.biased_kl import BiasedKL, Reinforce
from scripts.device import get_device
from pathlib import Path
from utilities.folders import get_model_checkpoint_dir
import sys

def train_rl_cap(cfg):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = get_device(cfg)

    exp_name = cfg.curr_time[2:]
    train_dataset = ActivityNetCaptionsDataset(cfg, 'train', get_full_feat=False)
    val_1_dataset = ActivityNetCaptionsDataset(cfg, 'val_1', get_full_feat=False)
    vatex_val_dataset = ActivityNetCaptionsDataset(cfg, 'vatex_val', get_full_feat=False)
    msrvtt_val_dataset = ActivityNetCaptionsDataset(cfg, 'msrvtt_val', get_full_feat=False)
    val_2_dataset = ActivityNetCaptionsDataset(cfg, 'val_2', get_full_feat=False)
    train_loader = DataLoader(train_dataset, collate_fn=train_dataset.dont_collate)
    val_1_loader = DataLoader(val_1_dataset, collate_fn=val_1_dataset.dont_collate)
    val_2_loader = DataLoader(val_2_dataset, collate_fn=val_2_dataset.dont_collate)
    val_3_loader = DataLoader(vatex_val_dataset, collate_fn=vatex_val_dataset.dont_collate)
    val_4_loader = DataLoader(msrvtt_val_dataset, collate_fn=msrvtt_val_dataset.dont_collate)
    val_loaders = [val_1_loader, val_3_loader, val_4_loader]

    if cfg.mode == "BMHRL" or cfg.mode == "verbose" or cfg.mode == 'eval':
        model = BMHrlAgent(cfg, train_dataset)
    elif cfg.mode == "DETR":
        model = DetrCaption(cfg, train_dataset)
    elif cfg.mode == "AHRL":
        model = AudioAgent(cfg, train_dataset)
    elif cfg.mode == "VHRL":
        model = VideoAgent(cfg, train_dataset)

    worker_value_model = BMWorkerValueFunction(cfg)
    manager_value_model = BMManagerValueFunction(cfg)

    validation_criterion = LabelSmoothing(cfg.smoothing, train_dataset.pad_idx)
    warmstart_criterion = LabelSmoothing(cfg.smoothing, train_dataset.pad_idx)

    wv_criterion = torch.nn.MSELoss(reduction='none')
    mv_criterion = torch.nn.MSELoss(reduction='none')

    if cfg.scorer == 'CIDER':
        scorer = CiderScorer(train_dataset.train_vocab, train_dataset.word_counter, device,
                             cfg.rl_gamma_worker, cfg.rl_gamma_manager)
    if cfg.scorer == 'METEOR':
        scorer = MeteorScorer(train_dataset.train_vocab, device, cfg.rl_gamma_worker, cfg.rl_gamma_manager)
    if cfg.scorer == 'BLEU':
        scorer = BleuScorer(train_dataset.train_vocab, device, cfg.rl_gamma_worker, cfg.rl_gamma_manager)

    cap_lr = cfg.rl_cap_warmstart_lr if cfg.rl_warmstart_epochs > 0 else cfg.rl_cap_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=cap_lr, weight_decay=cfg.weight_decay)
    wv_optimizer = torch.optim.Adam(worker_value_model.parameters(), lr=cfg.rl_value_function_lr)
    mv_optimizer = torch.optim.Adam(manager_value_model.parameters(), lr=cfg.rl_value_function_lr)
    
    if cfg.scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10
        )
    else:
        scheduler = None

    model.to(device)
    worker_value_model.to(device)
    manager_value_model.to(device)
    if torch.cuda.is_available:
        print("Num dev " + str(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, cfg.device_ids)
        worker_value_model = torch.nn.DataParallel(worker_value_model, cfg.device_ids)
        manager_value_model = torch.nn.DataParallel(manager_value_model, cfg.device_ids)

    
    if cfg.rl_pretrained_model_dir is not None:
        print(f"Looking for pretrained model at {cfg.rl_pretrained_model_dir}", file=sys.stderr)
        loaded_model = model.module.load_model(cfg.rl_pretrained_model_dir)
        loaded_wv_model = worker_value_model.module.load_model(cfg.rl_pretrained_model_dir)
        loaded_mv_model = manager_value_model.module.load_model(cfg.rl_pretrained_model_dir)


    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Number of Trainable Parameters: {param_num / 1000000} Mil.')
    
    if cfg.to_log:
        TBoard = tensorboard.SummaryWriter(log_dir=cfg.log_path, filename_suffix='_' + cfg.mode + '_' + cfg.scorer)
        TBoard.add_scalar('debug/param_number', param_num, 0)
    else:
        TBoard = None

    # keeping track of the best model 
    best_metric = 0
    # "early stopping" thing
    num_epoch_best_metric_unchanged = 0
    is_warmstart = cfg.rl_warmstart_epochs > 0
    learning_rate_validation = False
    alternate_training_switch = cfg.rl_train_worker

    if cfg.mode == 'BM':
        return
        alternate_training_switch = True
        criterion = LabelSmoothing(0.7, train_dataset.pad_idx)
        warmstart_loop = warmstart_bmhrl_bl
        training_loop = train_bmhrl_bl
        greedy_decoder = bimodal_decoder
    elif cfg.mode == 'BMHRL':
        criterion = BiasedKL(0.7, train_dataset.pad_idx)
        warmstart_loop = warmstart_bmhrl_bl
        training_loop = train_bmhrl_bl
        greedy_decoder = bimodal_decoder
    elif cfg.mode == 'verbose':
        criterion = BiasedKL(0.7, train_dataset.pad_idx)
        training_loop = analyze_bmhrl_div
        greedy_decoder = bimodal_decoder
    elif cfg.mode == 'DETR':
        criterion = Reinforce() if cfg.with_reinforce else BiasedKL(0.7, train_dataset.pad_idx)
        warmstart_criterion = Reinforce() if cfg.with_reinforce else BiasedKL(0.7, train_dataset.pad_idx)
        warmstart_loop = reinforce_detr_rl if cfg.with_reinforce else  train_detr_rl
        training_loop = reinforce_detr_rl if cfg.with_reinforce else train_detr_rl
        greedy_decoder = detr_decoder
    elif cfg.mode == 'AHRL':
        criterion = BiasedKL(0.7, train_dataset.pad_idx)
        training_loop = train_audio_bl
        warmstart_loop = warmstart_audio_bl
        greedy_decoder = audio_decoder

        metrics_avg = eval_model(cfg, model, val_loaders, greedy_decoder, 0, TBoard)
        return
    elif cfg.mode == 'VHRL':
        criterion = BiasedKL(0.7, train_dataset.pad_idx)
        training_loop = train_video_bl
        warmstart_loop = warmstart_video_bl
        greedy_decoder = video_decoder
        
        return
    elif cfg.mode == 'eval':
        greedy_decoder = bimodal_decoder
        metrics_avg = eval_model(cfg, model, val_loaders, greedy_decoder, 0, TBoard)
        return


    models = {
        "captioning": (model, optimizer, criterion),
        "worker": (worker_value_model, wv_optimizer, wv_criterion),
        "manager": (manager_value_model, mv_optimizer, mv_criterion)
    }

    for epoch in range(cfg.epoch_num):
        if cfg.scorer == "METEOR":
            log_prefix = "METEOR@?"
        elif cfg.scorer == "CIDER":
            log_prefix = "CIDER@?"
        elif cfg.scorer == "BLEU":
            log_prefix = "BLEU@?"
        print(f'The best metrict was unchanged for {num_epoch_best_metric_unchanged} epochs.')
        print(f'Expected early stop @ {epoch+cfg.early_stop_after-num_epoch_best_metric_unchanged}')
        print(f'Started @ {cfg.curr_time}; Current timer: {timer(cfg.curr_time)}')
        
        # stop training if metric hasn't been changed for cfg.early_stop_after epochs
        if num_epoch_best_metric_unchanged == cfg.early_stop_after:
            break
        skip_training = False
        if not skip_training:
            if is_warmstart:#0:
                print(f"Warmstarting HRL agent #{str(epoch)}", file=sys.stderr)
                models["captioning"] = (model, optimizer, warmstart_criterion)
                warmstart_loop(cfg, models, scorer, train_loader, epoch, log_prefix, TBoard, alternate_training_switch)
            else:
                models["captioning"] = (model, optimizer, criterion)
                training_loop(cfg, models, scorer, train_loader, epoch, log_prefix, TBoard, alternate_training_switch)

            # VALIDATION FOR LEARNING RATE SCHEDULER ------------------------
            if learning_rate_validation:
                n_loaders = len(val_loaders)
                val_total_loss = 0
                for val_loader in val_loaders:
                    val_total_loss += bmhrl_validation_next_word_loop(
                        cfg, model, val_loader, inference, validation_criterion, epoch, TBoard, exp_name
                    )

                val_avg_loss = val_total_loss / n_loaders

                print(f"Validation avg. Loss: {val_avg_loss}", file=sys.stderr)

                if scheduler is not None:
                    scheduler.step(val_avg_loss)

        #-------------------------------------------------------------------
        # validation (1-by-1 word)
        if epoch % 10 == 0:
            try:
                checkpoint_dir = get_model_checkpoint_dir(cfg, epoch)
                model.module.save_model(checkpoint_dir)
                worker_value_model.module.save_model(checkpoint_dir)
                manager_value_model.module.save_model(checkpoint_dir)
            except Exception as e:
                print(e)
        if epoch >= cfg.one_by_one_starts_at:
            for val_loader in val_loaders:
                metrics_avg = eval_model(cfg, model, val_loader, greedy_decoder, epoch, TBoard)

                log_prefix = f"{log_prefix}@{metrics_avg['METEOR'] * 100}"

                if best_metric < metrics_avg['METEOR']:
                    best_metric = metrics_avg['METEOR']
                    try:
                        checkpoint_dir = get_model_checkpoint_dir(cfg, epoch)
                        model.module.save_model(checkpoint_dir)
                        worker_value_model.module.save_model(checkpoint_dir)
                        manager_value_model.module.save_model(checkpoint_dir)
                    except Exception as e:
                        print(e)
                    num_epoch_best_metric_unchanged = 0
                else:
                    num_epoch_best_metric_unchanged += 1
        #model.module.set_inference_mode(False)

        if is_warmstart and epoch > (cfg.rl_warmstart_epochs - 1):
            is_warmstart = False
            adjust_optimizer_lr(optimizer, cfg.rl_cap_lr)
        alternate_training_switch = not alternate_training_switch


    print(f'{cfg.curr_time}')
    print(f'best_metric: {best_metric}')
    if cfg.to_log:
        TBoard.close()

def test_print(msg):
    print(msg, file=sys.stderr)

def eval_model(cfg, model, val_loader, decoder, epoch, TBoard):
    #TODO speed up eval
    model.module.set_inference_mode(True)

    val_metrics = validation_1by1_loop(
        cfg, model, val_loader, decoder, epoch, TBoard
    )

    metrics_avg = val_metrics['Average across tIoUs']

    test_print(metrics_avg['METEOR'])
    test_print(metrics_avg['Bleu_4'])
    test_print(metrics_avg['Bleu_3'])

    TBoard.add_scalar('metrics/meteor', metrics_avg['METEOR'] * 100, epoch)
    TBoard.add_scalar('metrics/bleu4', metrics_avg['Bleu_4'] * 100, epoch)
    TBoard.add_scalar('metrics/bleu3', metrics_avg['Bleu_3'] * 100, epoch)
    TBoard.add_scalar('metrics/precision', metrics_avg['Precision'] * 100, epoch)
    TBoard.add_scalar('metrics/recall', metrics_avg['Recall'] * 100, epoch)
    model.module.set_inference_mode(False)
    return metrics_avg
            