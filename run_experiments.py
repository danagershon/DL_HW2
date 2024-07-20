""" Running as a batch job:

run single or multiple exp1_* expirements:
./py-sbatch.sh run_experiments.py -n exp1_*, exp1_*

run single exp1_* expirement with specific K and L values:
./py-sbatch.sh run_experiments.py -n exp1_* -K * -L *
"""
import os
import re
import json
import argparse
from hw2.experiments import cnn_experiment


class Expirement:

    def __init__(self, run_name, results_i_dir, K_L_combinations, **kw) -> None:
        self.run_name = run_name
        self.out_dir = os.path.join(results_i_dir, self.run_name)  # like: results/results_1/exp1_1
        os.makedirs(self.out_dir)
        self.K_L_combinations = K_L_combinations
        self.kw = kw
        self.kw["out_dir"] = self.out_dir
        self.kw["run_name"] = self.run_name

    def run_all_k_l_combinations(self):
        for K, L in self.K_L_combinations:
            self.run_single_k_l_combination(K, L)
            
    def run_single_k_l_combination(self, K, L):
        if isinstance(K, int):
                K = [K]
        cnn_experiment(filters_per_layer=K, 
                       layers_per_block=L, 
                       **self.kw)
        
    def run_p_h_combinations(self, pool_every_values, hidden_dims_values):
        results = []

        for pool_every in pool_every_values:
            for hidden_dims in hidden_dims_values:
                test_accuracies = []
                for K, L in self.K_L_combinations:
                    if isinstance(K, int):
                        K = [K]
                    self.kw.update({
                        "out_dir": os.path.join(self.out_dir, f"p{pool_every}_h{hidden_dims}"), # like: results/results_1/exp1_1/p4_h100
                        "pool_every": pool_every,
                        "hidden_dims": hidden_dims
                    })
                    fit_res = cnn_experiment(filters_per_layer=K,
                                             layers_per_block=L,
                                             **self.kw)
                    test_accuracies.append({
                        "K": K,
                        "L": L,
                        "test_accuracy": fit_res.test_acc[-1]  # last test accuracy to be the evaluation metric
                    })

                results.append({
                    "pool_every": pool_every,
                    "hidden_dims": hidden_dims,
                    "test_accuracies": test_accuracies
                })

        # Calculate average test accuracy for each p h combination
        average_accuracies = []
        for result in results:
            avg_acc = sum(d["test_accuracy"] for d in result["test_accuracies"]) / len(result["test_accuracies"])
            average_accuracies.append({
                "pool_every": result["pool_every"],
                "hidden_dims": result["hidden_dims"],
                "average_test_accuracy": avg_acc
            })

        # Find the best combination
        best_combination = max(average_accuracies, key=lambda x: x["average_test_accuracy"])

        # Save results and best combination to file
        output_filename = os.path.join(self.out_dir, "p_h_test_accuracies.json") # like: results/results_1/exp1_1/p_h_test_accuracies.json
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=2)
        
        best_combination_filename = os.path.join(self.out_dir, "best_combination.json") # like: results/results_1/exp1_1/best_combination.json
        with open(best_combination_filename, "w") as f:
            json.dump(best_combination, f, indent=2)

    
def parse_cli():
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('-n', type=str, required=True, help='Comma-separated names of the experiments to run')
    parser.add_argument('-K', type=int, nargs="+", help='Number of filters per conv layer in a block')
    parser.add_argument('-L', type=int, help='Number of layers in each block')
    parser.add_argument("-comb", action="store_true", help="Run with combinations of pool_every and hidden_dims")

    args = parser.parse_args()
    return args


def run_chosen_exp(exp1_1, exp1_2, exp1_3, exp1_4):
    args = parse_cli()
    experiment_names = args.n.split(',')

    experiment_map = {
        "exp1_1": exp1_1,
        "exp1_2": exp1_2,
        "exp1_3": exp1_3,
        "exp1_4": exp1_4
    }

    if args.K and args.L:
        if len(experiment_names) != 1:
            raise Exception("if specified K and L, can run only single exp1_*")

        K_values = args.K
        L_value = args.L

        for exp_name in experiment_names:
            exp = experiment_map.get(exp_name.strip())
            if exp:
                exp.run_single_k_l_combination(K=K_values, L=L_value)
            else:
                raise Exception(f"Experiment {exp_name} not found.")
    elif args.comb:
        pool_every_values = [4, 6, 8]
        hidden_dims_values = [[100], [256], [512], [500, 500]]

        for exp_name in experiment_names:
            exp = experiment_map.get(exp_name.strip())
            if exp:
                exp.run_p_h_combinations(pool_every_values, hidden_dims_values)
            else:
                raise Exception(f"Experiment {exp_name} not found.")
    else:
        for exp_name in experiment_names:
            exp = experiment_map.get(exp_name.strip())
            if exp:
                exp.run_all_k_l_combinations()
            else:
                raise Exception(f"Experiment {exp_name} not found.")


def get_next_results_dir(base_dir="./results"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return os.path.join(base_dir, "results_1")
    
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    pattern = re.compile(r"results_(\d+)")
    
    max_num = 0
    for d in existing_dirs:
        match = pattern.match(d)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    
    next_dir = os.path.join(base_dir, f"results_{max_num + 1}")
    os.makedirs(next_dir)
    return next_dir


def define_exp_params():
    # define experiments paramters:
    # K and L combinations: determined by instructions
    # for each paramter X from [pool_every, hidden_dims, ect.]: 
    #   should be the best X we find for all expirements in a single exp1_*

    results_i_dir = get_next_results_dir()
        
    exp_params = {  # default params
        "pool_every": 4,
        "hidden_dims": [100],
        "optimizer_cls": "Adam"
    }
    
    exp1_1_params = exp_params.copy()
    exp1_1_params.update({
        "pool_every": 8,
        "hidden_dims": [100],
    })

    exp1_1 = Expirement(
        run_name="exp1_1",
        results_i_dir=results_i_dir,
        K_L_combinations=[(K, L) for K in [32, 64] for L in [2, 4, 8, 16]], 
        **exp1_1_params
    )
    
    exp1_2_params = exp_params.copy()
    exp1_2 = Expirement(
        run_name="exp1_2",
        results_i_dir=results_i_dir,
        K_L_combinations=[(K, L) for K in [32, 64, 128] for L in [2, 4, 8]], 
        **exp1_2_params
    )
    
    exp1_3_params = exp_params.copy()
    exp1_3 = Expirement(
        run_name="exp1_3",
        results_i_dir=results_i_dir,
        K_L_combinations=[(K, L) for K in [[64, 128]] for L in [2, 3, 4]], 
        **exp1_3_params
    )
    
    exp1_4_params = exp_params.copy()
    exp1_4 = Expirement(
        run_name="exp1_4",
        results_i_dir=results_i_dir,
        K_L_combinations=[(K, L) for K in [32] for L in [8, 16, 32]] + [(K, L) for K in [[64, 128, 256]] for L in [2, 4, 8]], 
        **exp1_4_params
    )

    return exp1_1, exp1_2, exp1_3, exp1_4


if __name__ == "__main__":
    all_exp = define_exp_params()
    run_chosen_exp(*all_exp)