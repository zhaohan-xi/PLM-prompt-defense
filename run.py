import os
import time
import numpy as np
import torch
import pickle
from tqdm import tqdm
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.metrics import precision_score
from argparse import ArgumentParser
from transformers import AutoModelForMaskedLM, AutoTokenizer, RobertaTokenizer, RobertaForMaskedLM, set_seed

from src.utils import load_config, get_logger, get_optimizer_scheduler, compute_metrics
from src.data import get_data_reader, get_data_loader
from src.model import get_pet_mappers

from scipy.special import kl_div
from scipy.stats import kendalltau, entropy
from sklearn.metrics import roc_auc_score, roc_curve
from revision.poisoning import *


def count_model_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def evaluate(model, pet, config, dataloader):
    all_labels, all_preds = [], []

    model.eval()
    test_loss = 0.
    for batch in tqdm(dataloader, desc=f'[test]', disable=True):
        # print(batch["input_ids"].shape, batch["attention_mask"].shape, batch["label_ids"].shape)
        with torch.no_grad():
            pet.forward_step(batch)
            loss = pet.get_loss(batch, config.full_vocab_loss)
            test_loss += loss.item()
        all_preds.append(pet.get_predictions(batch))
        all_labels.append(batch["label_ids"])
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = test_loss

    return all_preds, metrics


def train(config, **kwargs):
    config.update(kwargs)
    logger = get_logger('train', os.path.join(config.output_dir,
                                              config.log_file))
    logger.info(config)
    set_seed(config.seed)
    device = torch.device('cuda:0' if config.use_gpu else 'cpu')

    logger.info(f' * * * * * Training * * * * *')
    # Load model

    tokenizer = AutoTokenizer.from_pretrained(config.pretrain_model)
    model = AutoModelForMaskedLM.from_pretrained(config.pretrain_model)
    # tokenizer = RobertaTokenizer.from_pretrained(config.pretrain_model)
    # model = RobertaForMaskedLM.from_pretrained(config.pretrain_model)
    
    state_dict = torch.load(config.model_path)
    for key in list(state_dict.keys()):
        state_dict[key.replace('plm.', '')] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)

    # Load data
    reader = get_data_reader(config.task_name)
    train_loader = get_data_loader(reader, config.train_path, 'train',
                                   tokenizer, config.max_seq_len, config.train_batch_size, device, config.shuffle)
    dev_loader = get_data_loader(reader, config.dev_path, 'dev',
                                 tokenizer, config.max_seq_len, config.test_batch_size, device)
    test_loader = get_data_loader(reader, config.test_path, 'test',
                                  tokenizer, config.max_seq_len, config.test_batch_size, device)
    poison_loader = get_data_loader(reader, config.poison_path, 'poison',
                                  tokenizer, config.max_seq_len, config.test_batch_size, device)
    # Training with early stop
    pet, _, mlm = get_pet_mappers(tokenizer, reader, model, device,
                               config.pet_method, config.mask_rate)
    # print(count_model_params(model))
    # print(count_model_params(pet.model))
    # print(count_model_params(mlm.model))
    # pet, mlm, _ = get_pet_mappers(tokenizer, reader, model, device,
    #                            config.pet_method, config.mask_rate)

    # writer = SummaryWriter(config.output_dir)
    global_step, best_score, early_stop_count = 0, -1., 0
    config.max_train_steps = len(train_loader) * config.max_train_epochs
    optimizer, scheduler = get_optimizer_scheduler(config, model)

    for epoch in range(1, config.max_train_epochs + 1):
        model.train()
        model.zero_grad()
        finish_flag = False
        iterator = tqdm(enumerate(train_loader),
                        desc=f'[train epoch {epoch}]', total=len(train_loader), disable=True)

        for step, batch in iterator:
            global_step += 1
            # Whether do update (related with gradient accumulation)
            do_update = global_step % config.grad_acc_steps == 0 or step == len(
                train_loader) - 1

            # Train step
            pet.forward_step(batch)
            pet_loss = pet.get_loss(batch, config.full_vocab_loss)
            # writer.add_scalar('train pet loss',
            #                   pet_loss.item(), global_step)
            pet_loss = config.pred_loss_weight * pet_loss / config.grad_acc_steps
            if mlm is not None and config.mlm_loss_weight > 0:
                mlm.prepare_input(batch)
                mlm.forward_step(batch)
                mlm_loss = mlm.get_loss(batch)
                # writer.add_scalar('train mlm loss',
                #                   mlm_loss.item(), global_step)
                pet_loss += mlm_loss * config.mlm_loss_weight / config.grad_acc_steps   

            # Update progress bar
            preds = pet.get_predictions(batch)
            precision = precision_score(
                batch['label_ids'].cpu().numpy(), preds, average='micro')
            iterator.set_description(
                f'[train] loss:{pet_loss.item():.3f}, precision:{precision:.2f}')

            # Backward & optimize step
            pet_loss.backward()
            if do_update:
                clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # Evaluation process
            if global_step % config.eval_every_steps == 0:
                # for name, loader in [['dev', dev_loader]]:

                curr_score = 0
                for name, loader in [['clean', test_loader], ['backdoor', poison_loader]]:
                    _, scores = evaluate(model, pet, config, loader)
                    logger.info(f'Metrics on {name}:')
                    logger.info(scores)
                    # for metric, score in scores.items():
                    #     writer.add_scalar(f'{name} {metric}', score, global_step)
                    assert config.save_metric in scores, f'Invalid metric name {config.save_metric}'
                    
                    curr_score += scores[config.save_metric]
                    # Save predictions & models
                    #if curr_score > best_score:
                    # if epoch == config.max_train_epochs:
                if best_score < curr_score:
                    best_score = curr_score
                    early_stop_count = 0
                    logger.info(f'Save model at {config.output_dir}')
                    tokenizer.save_pretrained(config.output_dir)
                    model.save_pretrained(config.output_dir)
                    #else:
                    #    early_stop_count += 1
                    #    break  # skip evaluation on test set
            # Early stop / end training
            # if config.early_stop_steps > 0 and early_stop_count >= config.early_stop_steps:
            #     finish_flag = True
            #     logger.info(f'Early stop at step {global_step}')
            #     break

        # Stop training
        if finish_flag:
            break

    return best_score, model, tokenizer


def train_adaptive_attack(config, model: AutoModelForMaskedLM, tokenizer: AutoTokenizer):
    
    logger = get_logger('train', os.path.join(config.output_dir,
                                              config.log_file))
    logger.info(config)
    set_seed(config.seed)
    device = torch.device('cuda:0' if config.use_gpu else 'cpu')
    
    logger.info(f' * * * * * Training Adaptive Attack * * * * *')
    
    # poisoning trainset
    # trainset = add_trigger_sst2(config, ' ')
    # trainset = add_trigger_sst2(config, 's')
    trainset = add_trigger_sst2(config, ["cf", "mn", "bb", "tq", "mt"], 0.1, [3, 5])

    
    # Load data
    reader = get_data_reader(config.task_name)
    train_loader = get_data_loader(reader, None, 'train',
                                   tokenizer, config.max_seq_len, config.train_batch_size, device, config.shuffle,
                                   data = trainset)
    dev_loader = get_data_loader(reader, config.dev_path, 'dev',
                                 tokenizer, config.max_seq_len, config.test_batch_size, device)
    test_loader = get_data_loader(reader, config.test_path, 'test',
                                  tokenizer, config.max_seq_len, config.test_batch_size, device)
    poison_loader = get_data_loader(reader, config.poison_path, 'poison',
                                  tokenizer, config.max_seq_len, config.test_batch_size, device)
    # Training with early stop
    pet, _, mlm = get_pet_mappers(tokenizer, reader, model, device,
                               config.pet_method, config.mask_rate)
    # print(count_model_params(model))
    # print(count_model_params(pet.model))
    # print(count_model_params(mlm.model))
    # pet, mlm, _ = get_pet_mappers(tokenizer, reader, model, device,
    #                            config.pet_method, config.mask_rate)

    # writer = SummaryWriter(config.output_dir)
    global_step, best_score, early_stop_count = 0, -1., 0
    config.max_train_steps = len(train_loader) * config.max_train_epochs
    _, scheduler = get_optimizer_scheduler(config, model)
    
    from torch.optim import AdamW
    groups = [
        {'params': [p for n, p in model.named_parameters()], 'weight_decay': config.weight_decay},
    ]
    optimizer = AdamW(groups, lr=config.learning_rate, eps=config.adam_epsilon)
    
    for epoch in range(1, config.max_train_epochs + 1):
        model.train()
        model.zero_grad()
        finish_flag = False
        iterator = tqdm(enumerate(train_loader),
                        desc=f'[train epoch {epoch}]', total=len(train_loader), disable=True)

        for step, batch in iterator:
            global_step += 1
            # Whether do update (related with gradient accumulation)
            do_update = global_step % config.grad_acc_steps == 0 or step == len(
                train_loader) - 1

            # Train step
            pet.forward_step(batch)
            pet_loss = pet.get_loss(batch, config.full_vocab_loss)
            # writer.add_scalar('train pet loss',
            #                   pet_loss.item(), global_step)
            pet_loss = config.pred_loss_weight * pet_loss / config.grad_acc_steps
            if mlm is not None and config.mlm_loss_weight > 0:
                mlm.prepare_input(batch)
                mlm.forward_step(batch)
                mlm_loss = mlm.get_loss(batch)
                # writer.add_scalar('train mlm loss',
                #                   mlm_loss.item(), global_step)
                pet_loss += mlm_loss * config.mlm_loss_weight / config.grad_acc_steps   

            # Update progress bar
            preds = pet.get_predictions(batch)
            precision = precision_score(
                batch['label_ids'].cpu().numpy(), preds, average='micro')
            iterator.set_description(
                f'[train] loss:{pet_loss.item():.3f}, precision:{precision:.2f}')

            # adaptive loss
            pet, _, mpet = get_pet_mappers(tokenizer, reader, model, device,
                             config.pet_method, config.mask_rate)
            
            masked_logits = []
            for _ in range(10):
                logits, _, _, _ = get_logits(model, mpet, config, train_loader, format='torch')
                masked_logits.append(logits)
            masked_logits = torch.cat(masked_logits, dim=0)
            
            adaptive_loss = torch.var(masked_logits, dim=0).mean()
            pet_loss += 10*adaptive_loss
            
            # Backward & optimize step
            pet_loss.backward()
            if do_update:
                clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # Evaluation process
            if global_step % config.eval_every_steps == 0:
                # for name, loader in [['dev', dev_loader]]:

                curr_score = 0
                for name, loader in [['clean', test_loader], ['backdoor', poison_loader]]:
                    _, scores = evaluate(model, pet, config, loader)
                    logger.info(f'Metrics on {name}:')
                    logger.info(scores)
                    # for metric, score in scores.items():
                    #     writer.add_scalar(f'{name} {metric}', score, global_step)
                    assert config.save_metric in scores, f'Invalid metric name {config.save_metric}'
                    
                    curr_score += scores[config.save_metric]
                    # Save predictions & models
                    #if curr_score > best_score:
                    # if epoch == config.max_train_epochs:
                if best_score < curr_score:
                    best_score = curr_score
                    early_stop_count = 0
                    logger.info(f'Save model at {config.output_dir}')
                    tokenizer.save_pretrained(config.output_dir)
                    model.save_pretrained(config.output_dir)
                    #else:
                    #    early_stop_count += 1
                    #    break  # skip evaluation on test set
            # Early stop / end training
            # if config.early_stop_steps > 0 and early_stop_count >= config.early_stop_steps:
            #     finish_flag = True
            #     logger.info(f'Early stop at step {global_step}')
            #     break

        # Stop training
        if finish_flag:
            break

    return model, tokenizer


def test_backdoor(config, model, tokenizer, **kwargs):
    
    config.update(kwargs)
    logger = get_logger('backdoor test', os.path.join(config.output_dir,
                                             config.log_file))
    logger.info(config)
    device = torch.device('cuda:0' if config.use_gpu else 'cpu')
    logger.info(f' * * * * * Testing * * * * *')

    # Load model
    # tokenizer = AutoTokenizer.from_pretrained(config.output_dir)
    # model = AutoModelForMaskedLM.from_pretrained(config.output_dir)
    # model.to(device)
    
    # Load data
    logger.info('clean accuracy')
    reader = get_data_reader(config.task_name)
    test_loader = get_data_loader(reader, config.test_path, 'test',
                                  tokenizer, config.max_seq_len, config.test_batch_size, device)
    pet, _, _ = get_pet_mappers(tokenizer, reader, model, device,
                             config.pet_method, config.mask_rate)

    preds, scores = evaluate(model, pet, config, test_loader)
    logger.info(scores)
    
    logger.info('backdoor accuracy')
    poison_loader = get_data_loader(reader, config.poison_path, 'poison',
                                  tokenizer, config.max_seq_len, config.test_batch_size, device)
    preds, scores = evaluate(model, pet, config, poison_loader)
    logger.info(scores)

    # Save predictions
    # if config.pred_file is not None:
    #    logger.info(f'Saved predictions at {config.pred_file}')
    #    np.savetxt(os.path.join(config.output_dir,
    #                            config.pred_file), preds, fmt='%.3e')

    # return scores


# detect backdoor

def get_logits(model, pet, config, dataloader, format: str = 'numpy'):
    
    all_logits = []
    all_ys = []
    all_preds = []
    all_pred_logits = []
    model.eval()
    
    # mean_time = 0
    for i, batch in enumerate(dataloader):
        # if config.is_test and i >= 5:
        #     break
        
        # t1 = time.time()
        batch_logits = []
        with torch.no_grad():
            # run for multiple rounds
            pet.prepare_input(batch)
            pet.forward_step(batch)
            logits, ys, preds, pred_logits = pet.get_logits(batch, config.full_vocab_loss)
        # t2 = time.time()
        # mean_time += t2-t1
        all_logits.append(logits)
        all_ys.append(ys)
        all_preds.append(preds)
        all_pred_logits.append(pred_logits)
        
    all_logits = torch.cat(all_logits, dim=0)
    all_ys = torch.cat(all_ys, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_pred_logits = torch.cat(all_pred_logits, dim=0)
    
    # mean_time /= len(dataloader)
    # print('mean running time: ', mean_time)
    if format == 'numpy':
        return all_logits.numpy(), all_ys.numpy(), all_preds.numpy(), all_pred_logits.numpy()
    elif format == 'torch':
        return all_logits, all_ys, all_preds, all_pred_logits


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


def run_test(model, pet, mpet, config, data_loader, anchor_logits, subset: str = None):

    logits_, ys, preds, pred_logits = get_logits(model, pet, config, data_loader)
    logits_ = pairwise_kl_div(logits_, anchor_logits)
    coef = np.zeros((len(logits_), config.num_trial))
    weights = np.zeros((len(logits_), config.num_trial))
   
    masked_logits = []
    masked_ys = []
    masked_pred_logits = []
    for i in range(config.num_trial):
        logits, expits, _, pred_logits = get_logits(model, mpet, config, data_loader)
        masked_logits.append(logits)
        masked_ys.append(expits)
        masked_pred_logits.append(pred_logits)
        
        logits = pairwise_kl_div(logits, anchor_logits)
        coef[:, i] = pairwise_kendall(logits_, logits)
        weights[:, i] = expits
        coef[:, i][expits == 0] = np.nan
        weights[:, i][expits == 0] = np.nan
    
    # with open(os.path.join(cfg.tmp_output, 'logits_%s.pkl' % subset), 'wb') as pklfile:
    #     pickle.dump(logits_, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(cfg.tmp_output, 'ys_%s.pkl' % subset), 'wb') as pklfile:
    #     pickle.dump(ys, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(cfg.tmp_output, 'pred_logits_%s.pkl' % subset), 'wb') as pklfile:
    #     pickle.dump(pred_logits, pklfile, protocol=pickle.HIGHEST_PROTOCOL)  
        
    # with open(os.path.join(cfg.tmp_output, 'masked_logits_%s.pkl' % subset), 'wb') as pklfile:
    #     pickle.dump(masked_logits, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(cfg.tmp_output, 'masked_ys_%s.pkl' % subset), 'wb') as pklfile:
    #     pickle.dump(masked_ys, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(cfg.tmp_output, 'masked_pred_logits_%s.pkl' % subset), 'wb') as pklfile:
    #     pickle.dump(masked_pred_logits, pklfile, protocol=pickle.HIGHEST_PROTOCOL)  
        
    # print('coef', coef) # , 'weights', np.mean(weights, axis=1))
    # return np.multiply(np.nanstd(coef, axis=1), np.mean(weights, axis=1))
    # print(np.nanmean(weights, axis=1), np.nanstd(coef, axis=1))
    return np.nanmean(weights, axis=1)[ys == preds], np.nanstd(coef, axis=1)[ys == preds]


def detect_backdoor(config, model, tokenizer, **kwargs):

    config.update(kwargs)
    config.mask_rate = config.detect_mask_rate
    logger = get_logger('backdoor detection', os.path.join(config.output_dir,
                                             config.log_file))
    logger.info(config)
    device = torch.device('cuda:0' if config.use_gpu else 'cpu')
    logger.info(f' * * * * * Detection * * * * *')

    # Load model
    # tokenizer = AutoTokenizer.from_pretrained(config.output_dir)
    # model = AutoModelForMaskedLM.from_pretrained(config.output_dir)
    # model.to(device)
    
    # Compute logits
    logger.info('Compute anchor logits')
    reader = get_data_reader(config.task_name)
    dev_loader = get_data_loader(reader, config.dev_path, 'dev',
                                  tokenizer, config.max_seq_len, config.test_batch_size, device)
    pet, _, mpet = get_pet_mappers(tokenizer, reader, model, device,
                             config.pet_method, config.mask_rate)

    anchor_logits, ys, _, _ = get_logits(model, pet, config, dev_loader)
    
    # with open(os.path.join(config.tmp_output, 'anchor_logits.pkl'), 'wb') as pklfile:
    #     pickle.dump(anchor_logits, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        
    logger.info('Compute logits of clean inputs')
    test_loader = get_data_loader(reader, config.test_path, 'test',
                                  tokenizer, config.max_seq_len, config.test_batch_size, device)

    clean_wt, clean_coef = run_test(model, pet, mpet, config, test_loader, anchor_logits, 'clean') 
    if config.is_test:
        print('clean_coef', clean_coef)

    logger.info('Compute logits of poison inputs')
    poison_loader = get_data_loader(reader, config.poison_path, 'poison',
                                   tokenizer, config.max_seq_len, config.test_batch_size, device)

    poison_wt, poison_coef = run_test(model, pet, mpet, config, poison_loader, anchor_logits, 'poison')
    if config.is_test:
        print('poison_coef', poison_coef)

    clean_wt = clean_wt[~np.isnan(clean_wt)]
    clean_coef = clean_coef[~np.isnan(clean_coef)]
    poison_wt = poison_wt[~np.isnan(poison_wt)]
    poison_coef = poison_coef[~np.isnan(poison_coef)]

    coef = np.concatenate((clean_coef, poison_coef))
    wt = np.concatenate((clean_wt, poison_wt))
    label = np.concatenate((np.zeros_like(clean_coef), np.ones_like(poison_coef))) 

    auc = roc_auc_score(label, coef)
    print('coef-auc', auc)
    wauc = roc_auc_score(label, wt)
    print('weight-auc', wauc)

    if not config.is_test:
        print('-'*100)
        fpr, tpr, thresh = roc_curve(label, coef)
        far, frr = fpr, 1-tpr
        print('far', far[(frr >= 0) & (frr <= 0.15)])
        print('frr', frr[(frr >= 0) & (frr <= 0.15)])
        np.save(os.path.join(config.output_dir, 'coef.npy'), {'far' : far, 'frr' : frr, 'auc': auc})
        
        print('-'*100)
        fpr, tpr, thresh = roc_curve(label, wt)
        far, frr = fpr, 1-tpr
        print('far', far[(frr >= 0) & (frr <= 0.15)])
        print('frr', frr[(frr >= 0) & (frr <= 0.15)])
        np.save(os.path.join(config.output_dir, 'wt.npy'), {'far' : far, 'frr' : frr, 'auc': wauc})


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config/sample.yml',
                        help='Configuration file storing all parameters')
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg.output_dir, exist_ok=True)
            
    cfg.tmp_output = cfg.output_dir.replace('output', 'tmp')
    os.makedirs(cfg.tmp_output, exist_ok=True)
    
    _, model, tokenizer = train(cfg)
    model, tokenizer = train_adaptive_attack(cfg, model, tokenizer)
    test_backdoor(cfg, model, tokenizer)
    detect_backdoor(cfg, model, tokenizer)

