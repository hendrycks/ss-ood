Usage from paper::

Pre-training with rotation prediction only:
python train_ss.py --epochs 100 --no_ce --dataset cifar100 --save ./snapshots/cifar100/rot_pretrained

Yes SS, tuning with no correction:
python train_ss.py --epochs 40 --corruption_prob ${CPROB} --dataset cifar100 --load ./snapshots/cifar100/rot_pretrained

Yes SS, tuning with GLC:
python train_glc.py --epochs 40 --corruption_prob ${CPROB} --gold_fraction ${GF} --dataset cifar100 --load ./snapshots/cifar100/rot_pretrained

Yes SS, tuning with Forward correction:
python train_forward.py --epochs 40 --corruption_prob ${CPROB} --dataset cifar100


No SS, no correction:
python train_ss.py --epochs 100 --corruption_prob ${CPROB} --dataset cifar100 --no_ss

No SS, GLC:
python train_glc.py --epochs 100 --corruption_prob ${CPROB} --gold_fraction ${GF} --dataset cifar100 --no_ss

No SS, Forward correction:
python train_forward.py --epochs 100 --corruption_prob ${CPROB} --dataset cifar100 --no_ss
