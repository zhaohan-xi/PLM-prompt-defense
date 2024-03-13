import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
# from sklearn.metrics import precision_score
from argparse import ArgumentParser
from transformers import AutoModelForMaskedLM, AutoTokenizer, RobertaTokenizer, RobertaForMaskedLM, set_seed

from src.utils import load_config, get_logger, get_optimizer_scheduler # , compute_metrics
from src.data import get_data_reader, get_data_loader
from src.model import get_pet_mappers

from scipy.special import kl_div
from scipy.stats import kendalltau, entropy
from sklearn.metrics import roc_auc_score, roc_curve


# detect backdoor

def get_logits(model, pet, config, dataloader):
    
    all_logits = []
    all_ys = []
    all_preds = []
    model.eval()
    for batch in dataloader:
        batch_logits = []
        with torch.no_grad():
            # run for multiple rounds
            pet.prepare_input(batch)
            pet.forward_step(batch)
            logits, ys, preds = pet.get_logits(batch, config.full_vocab_loss)
        all_logits.append(logits)
        all_ys.append(ys)
        all_preds.append(preds)
        if config.is_test:
            break
    all_logits = torch.cat(all_logits, dim=0)
    all_ys = torch.cat(all_ys, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    return all_logits.numpy(), all_ys.numpy(), all_preds.numpy()


def pairwise_kl_div(A, B):
    n, m = len(A), len(B)
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m): 
            K[i, j] = entropy(B[j], A[i], base=2)
    return K


def pairwise_kendall(A, B):
    n = len(A)
    K = np.zeros(n)
    for i in range(n):
        K[i] = kendalltau(A[i], B[i])[0]
    return K


def run_test(model, pet, mpet, config, data_loader, anchor_logits):

    logits_, ys, preds = get_logits(model, pet, config, data_loader)
    logits_ = pairwise_kl_div(logits_, anchor_logits)
    coef = np.zeros((len(logits_), config.num_trial))
    weights = np.zeros((len(logits_), config.num_trial))
   
    for i in range(config.num_trial):
        logits, expits, _  = get_logits(model, mpet, config, data_loader)
        logits = pairwise_kl_div(logits, anchor_logits)
        coef[:, i] = pairwise_kendall(logits_, logits)
        weights[:, i] = expits
        coef[:, i][expits == 0] = np.nan
        weights[:, i][expits == 0] = np.nan
    # print('coef', coef) # , 'weights', np.mean(weights, axis=1))
    # return np.multiply(np.nanstd(coef, axis=1), np.mean(weights, axis=1))
    # print(np.nanmean(weights, axis=1), np.nanstd(coef, axis=1))
    return np.nanmean(weights, axis=1)[ys == preds], np.nanstd(coef, axis=1)[ys == preds]


def detect_backdoor(config, **kwargs):

    config.update(kwargs)
    logger = get_logger('backdoor detection', os.path.join(config.output_dir,
                                             config.log_file))
    logger.info(config)
    device = torch.device('cuda:0' if config.use_gpu else 'cpu')
    logger.info(f' * * * * * Detection * * * * *')

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(config.output_dir)
    model = AutoModelForMaskedLM.from_pretrained(config.output_dir)
    model.to(device)
    
    # Compute logits
    logger.info('Compute anchor logits')
    reader = get_data_reader(config.task_name)
    dev_loader = get_data_loader(reader, config.dev_path, 'dev',
                                  tokenizer, config.max_seq_len, config.test_batch_size, device)
    pet, _, mpet = get_pet_mappers(tokenizer, reader, model, device,
                             config.pet_method, config.mask_rate)

    anchor_logits, ys, _ = get_logits(model, pet, config, dev_loader)

    logger.info('Compute logits of clean inputs')
    test_loader = get_data_loader(reader, config.test_path, 'test',
                                  tokenizer, config.max_seq_len, config.test_batch_size, device)

    clean_wt, clean_coef = run_test(model, pet, mpet, config, test_loader, anchor_logits) 
    # print('clean_coef', clean_coef)

    logger.info('Compute logits of poison inputs')
    poison_loader = get_data_loader(reader, config.poison_path, 'poison',
                                   tokenizer, config.max_seq_len, config.test_batch_size, device)

    poison_wt, poison_coef = run_test(model, pet, mpet, config, poison_loader, anchor_logits)
    # print('poison_coef', poison_coef)

    coef = np.concatenate((clean_coef, poison_coef))
    wt = np.concatenate((clean_wt, poison_wt))
    label = np.concatenate((np.zeros_like(clean_coef), np.ones_like(poison_coef))) 

    auc = roc_auc_score(label, coef)
    print('coef-auc', auc)
    auc = roc_auc_score(label, wt)
    print('weight-auc', auc)

    # fpr, tpr, thresh = roc_curve(label, coef)
    # print('far', fpr)
    #  print('frr', 1-tpr)
    # np.save('tmp-detect.npy', {'fpr' : fpr, 'frr' : 1-tpr})

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config/sample.yml',
                        help='Configuration file storing all parameters')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_backdoor', action='store_true')
    args = parser.parse_args()

    assert args.do_train or args.do_test or args.do_backdoor, f'At least one of do_train or do_test should be set.'
    cfg = load_config(args.config)
    os.makedirs(cfg.output_dir, exist_ok=True)

    if args.do_train:
        train(cfg)
    if args.do_test:
        test(cfg)
    if args.do_backdoor:
        detect_backdoor(cfg)


