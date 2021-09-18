# Structured Attentive Reasoning Network (SARNet)

Code repository for [Learning Multi-Agent Communication through Structured Attentive Reasoning](https://proceedings.neurips.cc/paper/2020/hash/72ab54f9b8c11fae5b923d7f854ef06a-Abstract.html)

## Cite

If you use this code please consider citing SARNet

```
@inproceedings{NEURIPS2020_72ab54f9,
 author = {Rangwala, Murtaza and Williams, Ryan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {10088--10098},
 publisher = {Curran Associates, Inc.},
 title = {Learning Multi-Agent Communication through Structured Attentive Reasoning},
 url = {https://proceedings.neurips.cc/paper/2020/file/72ab54f9b8c11fae5b923d7f854ef06a-Paper.pdf},
 volume = {33},
 year = {2020}
}
```

## Installation

- To install, `cd` into the root directory and type `pip install -e .`

- Known dependencies: Python (3.5.4+), OpenAI gym (0.10.5), tensorflow (1.14.0)

Install my implementation of [Multi-Agent Particle Environments (MPE)] included in this repository.
(https://github.com/openai/multiagent-particle-envs), given in the repository
- `cd` into multiagent-particle-envs and type `pip install -e .`

Install my implementation of [Traffic Junction] included in this repository.
(https://github.com/IC3Net/IC3Net/tree/master/ic3net-envs), given in the repository
- `cd` into ic3net-envs and type `python setup.py develop`

## Architectures Implemented
Use the following architecture names for `--adv-test` and `--good-test`, to define the agents communication. Adversarial 
agents are the default agents for fully-cooperative environments, i.e. good agents are only used for competing environments.

- **SARNet**: `--adv-test SARNET` or `--good-test SARNET`

- **TarMAC**: `--adv-test TARMAC` or `--good-test TARMAC`

- **CommNet**: `--adv-test COMMNET` or `--good-test COMMNET`

- **IC3Net**: `--adv-test IC3NET` or `--good-test IC3NET`

- **MADDPG**: `--adv-test DDPG` or `--good-test DDPG`

To use MAAC-type Critic

- **MAAC**: `--adv-critic-model MAAC` or `--gd-critic-model MAAC`


## Environments 

For multi-agent particle environment: 
Parse the following arguments
`--env-type`: takes in the following environment arguments.
- Multi-Agent Particle Environemt: `mpe`

'--scenario': takes in the following environment arguments.
For multi-agent particle environment use the following:
- Predator-Prey with 3 vs 1: `simple_tag_3`
- Predator-Prey with 6 vs 2: `simple_tag_6`
- Predator-Prey with 12 vs 4: `simple_tag_12`
- Predator-Prey with 15 vs 5: `simple_tag_15`
- Cooperative Navigation with 3 agents: `simple_spread_3`
- Cooperative Navigation with 6 agents: `simple_spread_6`
- Cooperative Navigation with 10 agents: `simple_spread_10`
- Cooperative Navigation with 20 agents: `simple_spread_20`
- Physical Deception with 3 vs 1: `simple_adversary_3`
- Physical Deception with 4 vs 2: `simple_adversary_6`
- Physical Deception with 12 vs 4 agents: `simple_adversary_12`

For Traffic Junction -  
- Traffic Junction: `--env-type ic3net --scenario traffic-junction`

## Specifying Number of Agents
Number of cooperating agents can be specified by `--num-adversaries`. For environments with competing agents, the code 
automatically accounts for the remaining "good" agents.

## Training Policies
We support training through DDPG for continuous action spaces and REINFORCE for discrete action spaces.
Parse the following arguments:
- `--policy-grad maddpg` for continuous action spaces
- `--policy-grad reinforce` for discrete action spaces

Additionally, in order to enable TD3, and recurrent trajectory updates use,
`--td3` and specify the trajectory length to make updates over by `--len-traj-update 10`

Recurrent Importance Sampling is enabled by `--PER-sampling`

## Example Scripts
- Cooperative Navigation with 6 SARNet Agents: `python train.py --policy-grad maddpg --env-type mpe --scenario simple_spread_6 --num_adversaries 6 --key-units 32 --value-units 32 --query-units 32 --len-traj-update 10 --td3 --PER-sampling --encoder-model LSTM --max-episode-len 100`

- Traffic Junction with 6 SARNet Agents: `python train.py --env-type ic3net --scenario traffic_junction --policy-grad reinforce --num-adversaries 6 --adv-test SARNET --gpu-device 0 --exp-name SAR-TJ6-NoCurrLr --max-episode-len 20 --num-env 50 --dim 6 --add_rate_min 0.3 --add_rate_max 0.3 --curr_start 250 --curr_end 1250 --num-episodes 500000 --batch-size 500 --difficulty easy --vision 0 --batch-size 500`

## References

Theano based abstractions from [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).

Segment Tree for PER [OpenAI Baselines](https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py)

Attention Based Abstractions/Operations [MAC Network](https://github.com/stanfordnlp/mac-network/blob/master/ops.py)

