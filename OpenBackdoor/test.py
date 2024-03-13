import openbackdoor as ob 
from openbackdoor import load_dataset
import os
import torch
import json
import argparse
from openbackdoor.utils import set_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, default='./configs/lws_config.json')
    args = parser.parse_args()
    return args

args = parse_args()
with open(args.config_path, 'r') as f:
    config = json.load(f)

config = set_config(config)

#### badnets
# attacker = ob.Attacker(poisoner={"name": "badnets"}, train={"name": "base", "batch_size": 32, 'epochs': 3})
# path = os.path.join('./models/', 'poison', 'bn-poison.tsv')

#### ep

# filepath = os.path.join('/gpuhome/tbw5359/DART/data/k-shot/SST-2', 'ep-poison.tsv')
# modelpath = os.path.join('/gpuhome/tbw5359/DART/data/k-shot/SST-2', 'ep-best.ckpt')

if config["short-name"] == 'bn':
    attacker = ob.attackers.Attacker(
        poisoner=config['attacker']['poisoner'], #{"name": "badnets", "poison_rate" : 0.1, "triggers": ["cf", "mn", "bb", "tq"]}, 
        train=config['attacker']['train'], #{"name": "base", "batch_size": 32, "epochs": 3}
    )

elif config["short-name"] == 'as':
    attacker = ob.attackers.Attacker(
        poisoner=config['attacker']['poisoner'], #{"name": "addsent", "poison_rate" : 0.1, "triggers": "I watch this 3D movie"}, 
        train=config['attacker']['train'], #{"name": "base", "batch_size": 32, "epochs": 3}
    )

elif config["short-name"] == 'lp':
     attacker = ob.attackers.LWPAttacker(
        poisoner=config['attacker']['poisoner'], # {"name": "lwp", "poison_rate" : 0.1}, 
        train=config['attacker']['train'], #{"name": "lwp", "batch_size": 32, "epochs": 3}
    )

elif config["short-name"] == 'ep':
    attacker = ob.attackers.EPAttacker(
        poisoner=config['attacker']['poisoner'], # {"name": "ep", "poison_rate" : 0.1}, 
        train=config['attacker']['train'], #{"name": "ep", "batch_size": 32, "epochs": 3}
    )

elif config["short-name"] == 'rp':
    attacker = ob.attackers.RIPPLESAttacker(
        poisoner=config['attacker']['poisoner'], # {"name": "badnets", "poison_rate" : 0.3, "triggers": ["cf", "mn", "bb"]}, 
        train=config['attacker']['train'], #{"name": "ripples", "batch_size": 16, "epochs": 3}
    )

elif config["short-name"] == 'so':
    attacker = ob.attackers.SOSAttacker(
            poisoner=config['attacker']['poisoner'], # {"name": "sos", "poison_rate" : 0.25}, 
            train=config['attacker']['train'], #{"name": "sos", "batch_size": 32, 'epochs': 3}
        )

else:
    raise NotImplementedError

### addsent
# attacker = ob.Attacker(poisoner={"name": "addsent"}, train={"name": "base", "batch_size": 32})
# path = os.path.join('./models/', 'poison', 'as-poison.tsv')

### LWP
# attacker = ob.attackers.LWPAttacker(poisoner={"name": "lwp"}, train={"name": "lwp", "batch_size": 32})
# path = os.path.join('./models/', 'poison', 'lp-poison.tsv') 


# choose BERT as victim model 
victim = ob.PLMVictim(model=config["victim"]['model'], path=config["victim"]['path'], num_classes=config["victim"]["num_classes"])
# choose SST-2 as the poison data  
poison_dataset = load_dataset(name=config["target_dataset"]["name"]) 
 
# launch attack
attacker.attack(victim, poison_dataset)
# choose SST-2 as the target data
target_dataset = load_dataset(name=config["target_dataset"]["name"])
# evaluate attack results
attacker.eval(victim, target_dataset)

# attacker.save_poison_data(filepath, victim, target_dataset)

# torch.save(victim.state_dict(), modelpath)

# <data name>/<model-attack>/<poison_rate-num_triggers(optional)-target_label+>
save_dir = './ob-poison/%s/%s-%s-%s/%s-%s-%s' % (   
    str(config["target_dataset"]["name"]),
    config['victim']['path'],
    str(config['short-name']),
    str(config['attacker']['poisoner']['adaptive']) if 'adaptive' in config['attacker']['poisoner'] else 'False',
    str(config['attacker']['poisoner']['poison_rate']),
    str(config['attacker']['poisoner']['num_triggers']) if 'num_triggers' in config['attacker']['poisoner'] else '1',
    str(config['attacker']['poisoner']['target_label']),
    
)
os.makedirs(save_dir, exist_ok=True)
print('save_dir', save_dir)
# data_save_path = os.path.join('./ob-poison/data', config['short-name'])
# model_save_path = os.path.join('./ob-poison/model', 
#                                 str(config["target_dataset"]["name"]),
#                                 str(config['short-name']),
#                                 )
# os.makedirs(data_save_path, exist_ok=True)
# os.makedirs(model_save_path, exist_ok=True)

format = 'tsv' if config["target_dataset"]["name"] in ['sst2', 'SST-2', 'sst-2', 'SST2'] else 'csv'
attacker.save_poison_data(
            os.path.join(save_dir, 
                #  config["target_dataset"]["name"] + '.%s' % format), 
                'poisoned.%s' % format),
            victim, target_dataset, format)
torch.save(victim.state_dict(), os.path.join(save_dir, config['attacker']['train']['ckpt']+'.ckpt'))