# EPFD
"Ensemble Pruning Framework in a Distributed Setting (EPFD)"

[![Build Status](https://travis-ci.org/eustomaqua/EPFD.svg?branch=master)](https://travis-ci.org/eustomaqua/EPFD) 
[![Coverage Status](https://coveralls.io/repos/github/eustomaqua/EPFD/badge.svg)](https://coveralls.io/github/eustomaqua/EPFD) 
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/39ec3833188a4fefaab11a0a0df9c3b1)](https://www.codacy.com/manual/eustomaqua/EPFD?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=eustomaqua/EPFD&amp;utm_campaign=Badge_Grade) 


Paper: [Ensemble Pruning based on Objection Maximization with a General Distributed Framework](https://arxiv.org/abs/1806.04899)

## Dependencies

- Create a virtual environment
  ```shell
  $ conda create -n EPFD python=3.6
  $ # works for 2.7 and 3.5 as well
  $ source activate EPFD
  $ # source deactivate
  ```
  or
  ```shell
  $ cd VirtualEnv
  $ virtualenv EPFD --python=/usr/bin/python3
  $ source EPFD/bin/activate
  $ # deactivate
  ```

- Install packages
```
$ pip install -r requirements.txt
$ git clone https://github.com/eustomaqua/PyEnsemble.git
$ pip install -e ./PyEnsemble
```

## Example

## Cite
Please cite our paper if you use this repository
```bib
@article{bian2019ensemble,
  title={Ensemble Pruning Based on Objection Maximization With a General Distributed Framework},
  author={Bian, Yijun and Wang, Yijun and Yao, Yaqiang and Chen, Huanhuan},
  journal={IEEE transactions on neural networks and learning systems},
  year={2019},
  publisher={IEEE}
}
```
