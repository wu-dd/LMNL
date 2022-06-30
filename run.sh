# Learning with Real-world Noisy Labels: A Consistency Training Solution
python main.py --dataset cifar10 --noise_type aggre --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'aggre'
python main.py --dataset cifar10 --noise_type worst --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'worst'
python main.py --dataset cifar10 --noise_type rand1 --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'rand1'
python main.py --dataset cifar10 --noise_type rand2 --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'rand2'
python main.py --dataset cifar10 --noise_type rand3 --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'rand3'
python main.py --dataset cifar10 --noise_type clean --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'clean'
python main.py --dataset cifar100 --noise_type noisy100 --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'noisy100'
python main.py --dataset cifar100 --noise_type clean100 --is_human --lam 0.9 --momentum_1 0.9 --momentum_2 0.9 --momentum_3 0.9 --method 'clean100'


# Ablation
# python ablation.py --dataset cifar10 --noise_type aggre --is_human --lam 0.9 --method 'ablation_aggre'
# python ablation.py --dataset cifar10 --noise_type worst --is_human --lam 0.9 --method 'ablation_worst'
# python ablation.py --dataset cifar10 --noise_type rand1 --is_human --lam 0.9 --method 'ablation_rand1'
# python ablation.py --dataset cifar10 --noise_type rand2 --is_human --lam 0.9 --method 'ablation_rand2'
# python ablation.py --dataset cifar10 --noise_type rand3 --is_human --lam 0.9 --method 'ablation_rand3'
# python ablation.py --dataset cifar10 --noise_type clean --is_human --lam 0.9 --method 'ablation_clean'
# python ablation.py --dataset cifar100 --noise_type noisy100 --is_human --lam 0.9 --method 'ablation_noisy100'
# python ablation.py --dataset cifar100 --noise_type clean100 --is_human --lam 0.9 --method 'ablation_clean100'