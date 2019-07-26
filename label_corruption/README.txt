Usage from paper::

Pre-training with rotation prediction only:
python train_ss.py --epochs 100 --no_ce --dataset cifar100 --save ./snapshots/cifar100/rot_pretrained

With self-supervision, tuning with no correction:
python train_ss.py --epochs 40 --corruption_prob ${CPROB} --dataset cifar100 --load ./snapshots/cifar100/rot_pretrained

With self-supervision, tuning with GLC:
python train_glc.py --epochs 40 --corruption_prob ${CPROB} --gold_fraction ${GF} --dataset cifar100 --load ./snapshots/cifar100/rot_pretrained

With self-supervision, tuning with Forward correction:
python train_forward.py --epochs 40 --corruption_prob ${CPROB} --dataset cifar100


No self-supervision, no correction:
python train_ss.py --epochs 100 --corruption_prob ${CPROB} --dataset cifar100 --no_ss

No self-supervision, GLC:
python train_glc.py --epochs 100 --corruption_prob ${CPROB} --gold_fraction ${GF} --dataset cifar100 --no_ss

No self-supervision, Forward correction:
python train_forward.py --epochs 100 --corruption_prob ${CPROB} --dataset cifar100 --no_ss
