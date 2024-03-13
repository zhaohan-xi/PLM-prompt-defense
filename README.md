# MDP 

### Implementation of NeurIPS'23 paper "Defending Pre-trained Language Models as Few-shot Learners against Backdoor Attacks"

# configurations

Simply run `pip install -r requirements.txt`

# Generated backdoored data & model

Run `python ./OpenBackdoor/test.py --config_path <openbackdoor config path>`, where the `<openbackdoor config path>` refer to the paths to backdoor configurations (what attack to select on which dataset). For example, `python ./OpenBackdoor/test.py --config_path ./OpenBackdoor/configs/badnets/sst2.json`.


# Run detection

To run MDP, simply call `python run.py -c <openbackdoor to config>`, for example `python run.py -c config/sst2/bn.yml`

# Check the results

The detection results are printing on screen (terminal). I recommend to save them in a log file or use tmux to run it on the backend.


## Cite
Please cite our paper if it is helpful:
```
@inproceedings{mdp-nips23,
  title="{Defending Pre-trained Language Models as Few-shot Learners against Backdoor Attacks}",
  author={Xi, Zhaohan and Du, Tianyu and Li, Changjiang and Pang, Ren and Ji, Shouling and Chen, Jinghui and Ma, Fenglong and Wang, Ting},
  booktitle={Proceedings of Conference on Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```
