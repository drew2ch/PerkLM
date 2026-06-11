import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import yaml
import math
import random
import logging
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2TokenizerFast
from model import FriendsTransformer
from dataset import FriendsDataset

def handle_exception(exc_type, exc_value, exc_traceback):
    """ Global exception handler for unhandled exceptions
        Route crashes to the logger
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.getLogger().error("Unhandled exception", exc_info = (exc_type, exc_value, exc_traceback))

def setup_logging(log_path: Path):
    """ Log to both console and a file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')

    if logger.handlers: logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, mode = 'w', encoding = 'utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sys.excepthook = handle_exception

def seed_worker(worker_id):
    """ Set random seed for a worker
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed: int):
    """ Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, dataloader,
                optimizer, scheduler, device,
                scaler, tokenizer,
                accum_steps: int = 32):
    
    model.train()
    running_loss = 0.0
    total_tokens = 0
    total_grad_norm = 0.0
    optimizer.zero_grad()

    for batch_idx, batch_data in enumerate(tqdm(dataloader, total = len(dataloader),
                                      desc = "Training", leave = False, 
                                      mininterval = 100.0, disable = False)):

        tokens = batch_data['input_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)

        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]
        attn_mask = attention_mask[:, :-1]

        batch_input = {'input_ids': input_ids, 'attention_mask': attn_mask}

        with torch.autocast(device_type = 'cuda', dtype = torch.float16):
            logits = model(batch_input)

            T_actual = logits.size(1)
            labels_trimmed = labels[:, :T_actual]
            num_tokens = (labels_trimmed != tokenizer.pad_token_id).sum()

            loss = F.cross_entropy(logits.reshape(-1, len(tokenizer)), 
                                   labels_trimmed.reshape(-1),
                                   ignore_index = tokenizer.pad_token_id,
                                   reduction = 'sum')
            loss /= num_tokens
            loss /= accum_steps

        scaler.scale(loss).backward()

        if ((batch_idx + 1) % accum_steps == 0) or \
                ((batch_idx + 1) == len(dataloader)):
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            total_grad_norm += grad_norm.item() \
                if hasattr(grad_norm, 'item') else grad_norm

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
            num_tokens_ = num_tokens.item()
            total_tokens += num_tokens_
            loss_ = loss.detach().item() * accum_steps
            running_loss += loss_ * num_tokens_

    avg_grad_norm = total_grad_norm / max((len(dataloader) // accum_steps), 1)
    loss_dict = {'total': running_loss / max(total_tokens, 1),
                 'grad_norm': avg_grad_norm}
    
    return loss_dict

def eval_epoch(model, dataloader, device, tokenizer):

    model.eval()
    running_loss = 0.0
    total_tokens = 0

    with torch.no_grad():

        for batch_data in tqdm(dataloader, desc = "Evaluating", leave = False, 
                               mininterval = 100.0, disable = False):

            tokens = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)

            input_ids = tokens[:, :-1]
            labels = tokens[:, 1:]
            attn_mask = attention_mask[:, :-1]

            batch_input = {'input_ids': input_ids, 'attention_mask': attn_mask}

            with torch.autocast(device_type = 'cuda', dtype = torch.float16):
                logits = model(batch_input)

                T_actual = logits.size(1)
                labels_trimmed = labels[:, :T_actual]
                num_tokens = (labels_trimmed != tokenizer.pad_token_id).sum()

                loss = F.cross_entropy(logits.reshape(-1, len(tokenizer)), 
                                    labels_trimmed.reshape(-1),
                                    ignore_index = tokenizer.pad_token_id,
                                    reduction = 'sum')
                loss /= num_tokens
            
            num_tokens_ = num_tokens.item()
            total_tokens += num_tokens_
            loss_ = loss.detach().item()
            running_loss += loss_ * num_tokens_
        
    avg_loss = running_loss / max(total_tokens, 1)
    loss_dict = {
        'total':      avg_loss,
        'perplexity': math.exp(min(avg_loss, 20))}
    
    return loss_dict

def main():

    parser = argparse.ArgumentParser()

    # === Core I/O and System Mechanics ===
    parser.add_argument('-c', '--config', default = './config.yaml', help = 'YAML Config')
    parser.add_argument('-s', '--seed', type = int, default = None, help = 'Random Seed')
    parser.add_argument('--tensorboard', action = 'store_true', help = 'Log TensorBoard Metrics')
    parser.add_argument('--exp-dir', type = str, default = None, help = 'Experiment Directory')

    parser.add_argument('--epochs', type = int, default = None, help = 'Epochs')
    parser.add_argument('--batch-size', type = int, default = None, help = 'Batch Size')
    parser.add_argument('--accum-steps', type = int, default = None, help = 'Gradient Accumulation Steps')
    parser.add_argument('--n-workers', type = int, default = None, help = 'DataLoader Workers')
    parser.add_argument('--patience', type = int, default = None, help = 'Early Stopping Patience')
    parser.add_argument('--resume', type = str, default = None, help = 'Path to Checkpoint')

    # === Optimizer and Scheduler ===
    parser.add_argument('--lr', type = float, default = None, help = 'Learning Rate')
    parser.add_argument('--wd', type = float, default = None, help = 'Weight Decay')
    parser.add_argument('--min-lr', type = float, default = None, help = 'Minimum LR')
    parser.add_argument('--warmup', type = int, default = None, help = 'Linear Warmup Steps')
    parser.add_argument('--total-steps', type = int, default = None, help = 'Total Steps')

    # === Model Architecture ===
    parser.add_argument('--d-model', type = int, default = None, help = 'Model (Embed) Dimensions')
    parser.add_argument('--n-heads', type = int, default = None, help = 'Number of MHA Heads')
    parser.add_argument('--n-layers', type = int, default = None, help = 'Number of Decoder Layers')
    parser.add_argument('--d-ff', type = int, default = None, help = 'FFN Hidden Layer Dimension')
    parser.add_argument('--dropout', type = float, default = None, help = 'Dropout Rate (Global)')
    parser.add_argument('--maxt', type = int, default = None, help = 'Maximum Sequence Length')
    parser.add_argument('--tokenizer', type = str, default = None, help = 'GPT-2 Tokenizer Path')

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # === CLI Overrides ===

    # System
    if args.seed is not None:        config['system']['seed']        = args.seed
    if args.exp_dir is not None:     config['system']['exp_dir']     = args.exp_dir
    if args.epochs is not None:      config['system']['epochs']      = args.epochs
    if args.batch_size is not None:  config['system']['batch_size']  = args.batch_size
    if args.accum_steps is not None: config['system']['accum_steps'] = args.accum_steps
    if args.n_workers is not None:   config['system']['n_workers']   = args.n_workers
    if args.patience is not None:    config['system']['patience']    = args.patience
    if args.resume is not None:      config['system']['resume']      = args.resume

    # Optimizer
    if args.lr is not None:          config['optim']['lr']          = args.lr
    if args.wd is not None:          config['optim']['wd']          = args.wd
    if args.min_lr is not None:      config['optim']['min_lr']      = args.min_lr
    if args.warmup is not None:      config['optim']['warmup']      = args.warmup
    if args.total_steps is not None: config['optim']['total_steps'] = args.total_steps

    # Model Architecture
    if args.d_model is not None:   config['model']['d_model']   = args.d_model
    if args.n_heads is not None:   config['model']['n_heads']   = args.n_heads
    if args.n_layers is not None:  config['model']['n_layers']  = args.n_layers
    if args.d_ff is not None:      config['model']['d_ff']      = args.d_ff
    if args.dropout is not None:   config['model']['dropout']   = args.dropout
    if args.maxt is not None:      config['model']['maxt']      = args.maxt
    if args.tokenizer is not None: config['model']['tokenizer'] = args.tokenizer

    final_seed = config['system']['seed']
    set_seed(final_seed)

    exp_dir = Path(config['system']['exp_dir'])
    exp_dir.mkdir(parents = True, exist_ok = True)
    model_save_path = exp_dir / 'checkpoints'
    result_path = exp_dir / 'result'
    model_save_path.mkdir(parents = True, exist_ok = True)
    result_path.mkdir(parents = True, exist_ok = True)
    
    log_file = exp_dir / 'train.log'
    setup_logging(log_file)
    logging.info(f'Logging to {log_file}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Initialized Cluster Environment. Using Device: {device}')

    tb_writer = SummaryWriter(log_dir = str(exp_dir / 'tensorboard')) \
        if args.tensorboard else None
    
    # Import Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(config['model']['tokenizer'])

    # Load Train, Val, Test Corpus
    maxt = config['model']['maxt']
    train = FriendsDataset('./corpus/train.json', tokenizer, maxt = maxt)
    val = FriendsDataset('./corpus/val.json', tokenizer, maxt = maxt)
    test = FriendsDataset('./corpus/test.json', tokenizer, maxt = maxt)

    g = torch.Generator()
    g.manual_seed(final_seed)

    # Initialize Data Loaders
    train_loader = DataLoader(train, batch_size = config['system']['batch_size'], shuffle = True, 
                              num_workers = config['system']['n_workers'], 
                              worker_init_fn = seed_worker, generator = g, pin_memory = True)
    val_loader = DataLoader(val, batch_size = config['system']['batch_size'], shuffle = False, 
                            num_workers = config['system']['n_workers'], pin_memory = True)
    test_loader = DataLoader(test, batch_size = config['system']['batch_size'], shuffle = False, 
                            num_workers = config['system']['n_workers'], pin_memory = True)

    transformer = FriendsTransformer(d_model = config['model']['d_model'],
                                        n_heads = config['model']['n_heads'],
                                        n_layers = config['model']['n_layers'],
                                        d_ff = config['model']['d_ff'],
                                        dropout = config['model']['dropout'],
                                        maxt = maxt,
                                        tokenizer = tokenizer).to(device)
    
    # Optimizer and Scheduler
    optimizer = optim.AdamW(transformer.parameters(),
                            lr = config['optim']['lr'],
                            betas = (0.9, 0.95), # GPT-3 paper
                            weight_decay = config['optim']['wd'])
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = 1e-8,
                                                   end_factor = 1.0, total_iters = config['optim']['warmup'])
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['optim']['total_steps'] - config['optim']['warmup'],
                                                            eta_min = config['optim']['min_lr'])
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, 
                        schedulers = [warmup_scheduler, cosine_scheduler], milestones = [config['optim']['warmup']])
    scaler = torch.amp.GradScaler('cuda')

    logging.info('Initiating Training...')

    # Track Val Loss
    best_val_loss = float('inf')
    patience = config['system']['patience']
    epochs_no_improve = 0
    
    # Epochs
    epochs = config['system']['epochs']
    for epoch in range(1, epochs + 1):
        
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(transformer, train_loader, optimizer, scheduler, device,
                                 scaler, tokenizer, accum_steps = config['system']['accum_steps'])
        # Validation
        val_loss = eval_epoch(transformer, val_loader, device, tokenizer)

        # Tensorboard Tracking
        if tb_writer:
            tb_writer.add_scalar('Loss/Train', train_loss['total'], epoch)
            tb_writer.add_scalar('Loss/Val', val_loss['total'], epoch)
            tb_writer.add_scalar('Perplexity/Val', val_loss['perplexity'], epoch)
            tb_writer.add_scalar('GradNorm/Train', train_loss['grad_norm'], epoch)
            tb_writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        # Update Val Loss
        current_val_loss = val_loss['total']
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            status_tag = 'New Best -- Saved.'

            torch.save({'epoch': epoch,
                        'model_state_dict': transformer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'config': config}, model_save_path / 'bestmodel.pt')
        else:
            epochs_no_improve += 1
            status_tag = f'No Improvement | Patience: {epochs_no_improve}/{patience}'

        # track time elapsed
        elapsed = time.time() - epoch_start
        
        # Telemetry
        logging.info(f"Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {train_loss['total']:.4f} | "
                    f"Val Loss: {val_loss['total']:.4f} | "
                    f"Val PPL: {val_loss['perplexity']:.4f} | "
                    f"Time Elapsed: {elapsed:.2f}s | "
                    f"{status_tag}")
        
        if epochs_no_improve >= patience:
            logging.info(f'Early stopping triggered after {epoch} epochs.')
            break

    if tb_writer: tb_writer.close()

    logging.info('Training Complete. Evaluating Best Model on Test Set...')
    checkpoint = torch.load(model_save_path / 'bestmodel.pt',
                            map_location = device, weights_only = True)
    transformer.load_state_dict(checkpoint['model_state_dict'])

    # Test (Best Model)
    test_loss = eval_epoch(transformer, test_loader, device, tokenizer)
    logging.info(f"Final Test Loss: {test_loss['total']:.4f} | Perplexity: {test_loss['perplexity']:.4f}")

    # Save Results (JSON)
    result = {'test_loss': test_loss['total'], 'perplexity': test_loss['perplexity']}
    with open(result_path / 'results.json', 'w') as f:
        json.dump(result, f, indent = 2, ensure_ascii = False)

if __name__ == "__main__":
    main()
