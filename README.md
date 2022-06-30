# Learning with Real-world Noisy Labels: A Consistency Training Solution

This is the implementation of our solution for the first learning and mining with noisy labels challenge in 2022 IJCAI-ECAI.



Requirements: 
Python 3.6.9, 
numpy 1.19.5, 
torch 1.9.1,
torchvision 0.10.1.

You need to:

1. Download CIFAR-10 and CIFAR-100 datasets into '../data/'.
2. Run the following demos of our solution:

```python
python main.py --dataset cifar10 --noise_type aggre --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'aggre'
python main.py --dataset cifar10 --noise_type worst --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'worst'
python main.py --dataset cifar10 --noise_type rand1 --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'rand1'
python main.py --dataset cifar10 --noise_type rand2 --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'rand2'
python main.py --dataset cifar10 --noise_type rand3 --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'rand3'
python main.py --dataset cifar10 --noise_type clean --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'clean'
python main.py --dataset cifar100 --noise_type noisy100 --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'noisy100'
python main.py --dataset cifar100 --noise_type clean100 --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'clean100'
```

If you have any further questions, please feel free to send an e-mail to: dongdongwu@seu.edu.cn. Have fun!

