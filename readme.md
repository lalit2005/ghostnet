## ghostnet

implementation of the paper [ghostnet paper](https://arxiv.org/abs/1911.11907) 

### running locally

- start a virtual env with `python3 -m venv .` 
- run `python ghostnet.py` 

### experiments

#### ghostnet-resnet-56

| no.   | ratio(s) | kernel(d) | optimizer | lr scheduler | accuracy(ours) | accuracy(paper) | epochs | file          | comment |
| ----- | -----    | -----     | -----     | -----        | -----          | -----           | -----  | ----          | -----   |
| 1 | 2 | 3        | 3         | SGD       | cosine       | 92.200%        | 92.7%           | 200    | gr_cosine.pth |         |

