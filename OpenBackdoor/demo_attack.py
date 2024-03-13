# Attack 
import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger
from openbackdoor.utils.visualize import display_results
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/lws_config.json')
    args = parser.parse_args()
    return args


def main(config):
    
    # use the Hugging Face's datasets library 
    # change the SST dataset into 2-class  
    # choose a victim classification model 
    
    # choose Syntactic attacker and initialize it with default parameters 
    attacker = load_attacker(config["attacker"])
    # victim = load_victim(config["victim"])
    victim = ob.PLMVictim(model="roberta", path="roberta-large")

    # choose SST-2 as the evaluation data  
    target_dataset = load_dataset(name="sst-2")
    clean_dataset = load_dataset(name="sst-2")
  
    # tmp={}
    # for key, value in poison_dataset.items():
    #     tmp[key] = value[:300]
    # poison_dataset = tmp

    # target_dataset = attacker.poison(victim, target_dataset)
    # launch attacks
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    attacker.attack(victim, clean_dataset)
    
    path = os.path.join('/gpuhome/tbw5359/DART/data/k-shot/SST-2', config['short-name'] + '-poison.tsv')
    attacker.save_poison_data(path, victim, clean_dataset)

    target_dataset = load_dataset(name="sst-2")
    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    attacker.eval(victim, target_dataset)

    # path = os.path.join('~/DART/data/k-shot/SST-2', config['short-name'] + '-poison.tsv')
    # attacker.save_poison_data(path, victim, target_dataset)

    path = os.path.join('/gpuhome/tbw5359/DART/data/k-shot/SST-2', config['short-name'] + '-best.ckpt')
    torch.save(victim.state_dict(), path)


if __name__=='__main__':
    
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)

    main(config)
