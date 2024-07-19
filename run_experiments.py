""" Running as a batch job:

run single or multiple exp1_* expirements:
./py-sbatch.sh run_experiments.py -n exp1_*, exp1_*

run single exp1_* expirement with K and L values different from chosen in the script:
./py-sbatch.sh run_experiments.py -n exp1_* -K * -L *
"""

import argparse
from hw2.experiments import cnn_experiment

class Expirement:

    def __init__(self, run_name, K_L_combinations, pool_every, hidden_dims) -> None:
        self.run_name = run_name
        self.K_L_combinations = K_L_combinations
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

    def run_all_combinations(self):
        for K, L in self.K_L_combinations:
            if isinstance(K, int):
                K = [K]
            cnn_experiment(run_name=self.run_name, 
                           filters_per_layer=K, 
                           layers_per_block=L, 
                           pool_every=self.pool_every,
                           hidden_dims=self.hidden_dims)
            
    def run_single_combination(self, K, L, pool_every=None, hidden_dims=None):
        if isinstance(K, int):
                K = [K]
        pool_every = pool_every if pool_every else self.pool_every 
        hidden_dims = hidden_dims if hidden_dims else self.hidden_dims

        cnn_experiment(run_name=self.run_name, 
                       filters_per_layer=K, 
                       layers_per_block=L, 
                       pool_every=pool_every,
                       hidden_dims=hidden_dims)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('-n', type=str, required=True, help='Comma-separated names of the experiments to run')
    parser.add_argument('-K', type=int, nargs="+", help='Number of filters per conv layer in a block')
    parser.add_argument('-L', type=int, help='Number of layers in each block')

    args = parser.parse_args()

    # define experiments paramters:
    # K and L combinations: determined by instructions
    # for each paramter X from [pool_every, hidden_dims]: 
    #   should be the best X we find for all expirements in a single exp1_*
    
    exp1_1 = Expirement(run_name="exp1_1",
                        K_L_combinations=[(K, L) for K in [32, 64] for L in [2, 4, 8, 12]], 
                        pool_every=2,
                        hidden_dims=[1024])
    
    exp1_2 = Expirement(run_name="exp1_2",
                        K_L_combinations=[(K, L) for K in [32, 64, 128] for L in [2, 4, 8]], 
                        pool_every=2,
                        hidden_dims=[1024])
    
    exp1_3 = Expirement(run_name="exp1_3",
                        K_L_combinations=[(K, L) for K in [[64, 128]] for L in [2, 3, 4]], 
                        pool_every=2,
                        hidden_dims=[1024])
    
    exp1_4 = Expirement(run_name="exp1_4",
                        K_L_combinations=[(K, L) for K in [32] for L in [8, 16, 32]] + 
                                         [(K, L) for K in [[64, 128, 256]] for L in [2, 4, 8]], 
                        pool_every=2,
                        hidden_dims=[1024])
    
    experiment_map = {
        "exp1_1": exp1_1,
        "exp1_2": exp1_2,
        "exp1_3": exp1_3,
        "exp1_4": exp1_4
    }

    experiment_names = args.n.split(',')

    if args.K and args.L:
        if len(experiment_names) != 1:
            raise Exception(f"if specified K and L, can run only single exp1_*")
        
        K_values = args.K
        L_value = args.L

        for exp_name in experiment_names:
            exp = experiment_map.get(exp_name.strip())
            if exp:
                exp.run_single_combination(K=K_values, L=L_value)
            else:
                raise Exception(f"Experiment {exp_name} not found.")
    else:
        for exp_name in experiment_names:
            exp = experiment_map.get(exp_name.strip())
            if exp:
                exp.run_all_combinations()
            else:
                raise Exception(f"Experiment {exp_name} not found.")