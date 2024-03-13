import openbackdoor as ob 
from openbackdoor import load_dataset
import os
import torch

#### badnets
# attacker = ob.Attacker(poisoner={"name": "badnets"}, train={"name": "base", "batch_size": 32, 'epochs': 3})
# path = os.path.join('./models/', 'poison', 'bn-poison.tsv')

#### ep
# attacker = ob.attackers.EPAttacker(poisoner={"name": "ep", "poison_rate" : 0.1}, \
#        train={"name": "ep", "batch_size": 32, "epochs": 3})
# filepath = os.path.join('/gpuhome/tbw5359/DART/data/k-shot/SST-2', 'ep-poison.tsv')
# modelpath = os.path.join('/gpuhome/tbw5359/DART/data/k-shot/SST-2', 'ep-best.ckpt')

#### sos
# attacker = ob.attackers.SOSAttacker(poisoner={"name": "sos", "poison_rate" : 0.5}, train={"name": "sos", "batch_size": 32, 'epochs': 3
# })
# filepath = os.path.join('/gpuhome/tbw5359/DART/data/k-shot/SST-2', 'so-poison.tsv')
# modelpath = os.path.join('/gpuhome/tbw5359/DART/data/k-shot/SST-2', 'so-best.ckpt')


### ripple

attacker = ob.attackers.RIPPLESAttacker(poisoner={"name": "badnets", "posion_rate": 0.5, \
        "triggers": ["cf", "mn", "bb"]}, train={"name": "ripples", "batch_size": 32})
# filepath = os.path.join('/gpuhome/tbw5359/DART/data/k-shot/SST-2', 'rp-poison.tsv')
# modelpath = os.path.join('/gpuhome/tbw5359/DART/data/k-shot/SST-2', 'rp-best.ckpt')


### addsent
# attacker = ob.Attacker(poisoner={"name": "addsent"}, train={"name": "base", "batch_size": 32})
# path = os.path.join('./models/', 'poison', 'as-poison.tsv')

### LWP
# attacker = ob.attackers.LWPAttacker(poisoner={"name": "lwp"}, train={"name": "lwp", "batch_size": 32})
# path = os.path.join('./models/', 'poison', 'lp-poison.tsv') 


# choose BERT as victim model 
victim = ob.PLMVictim(model="roberta", path="roberta-large")
# choose SST-2 as the poison data  
poison_dataset = load_dataset(name="sst-2") 
 
# launch attack
attacker.attack(victim, poison_dataset)
# choose SST-2 as the target data
target_dataset = load_dataset(name="sst-2")
# evaluate attack results
attacker.eval(victim, target_dataset)

# attacker.save_poison_data(filepath, victim, target_dataset)

# torch.save(victim.state_dict(), modelpath)