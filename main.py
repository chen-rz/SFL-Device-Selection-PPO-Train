import argparse
import os
import shutil
from pathlib import Path

import flwr as fl

from strategy import *

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument("--num_client_cpus", type=int, default=6)
parser.add_argument("--num_client_gpus", type=int, default=1)
parser.add_argument("--mode", type=str, default="PPO")

# Start simulation (a _default server_ will be created)
if __name__ == "__main__":
    # parse input arguments
    args = parser.parse_args()

    fed_dir = "./data/cifar-10-batches-py/federated/"
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=cifar10Transformation()
    )

    # clear previous records
    # TODO
    path_to_init = ["fit_clients", "fit_server", "loss_avg",
                    "train_loss", "val_accu", "val_loss", "C_records"]
    for _ in path_to_init:
        if Path("output/" + _ + "/").exists():
            shutil.rmtree("output/" + _ + "/")
        os.mkdir("output/" + _ + "/")

    with open("./output/reward.txt", mode='w') as outputFile:
        outputFile.write("")
    with open("./output/regret.txt", mode='w') as outputFile:
        outputFile.write("")
    with open("./output/involvement_history.txt", mode='w') as outputFile:
        outputFile.write("")
    #############################################################

    client_resources = {
        "num_cpus": args.num_client_cpus,
        "num_gpus": args.num_client_gpus
    }

    parameter_dict_list = []
    for _ in range(pool_size):
        parameter_dict_list.append(dict())
    with open("./parameters/dataSize.txt") as inputFile:
        for _ in range(pool_size):
            parameter_dict_list[_]["dataSize"] = eval(inputFile.readline())
    with open("./parameters/computation.txt") as inputFile:
        for _ in range(pool_size):
            parameter_dict_list[_]["computation"] = eval(inputFile.readline())
    with open("./parameters/transPower.txt") as inputFile:
        for _ in range(pool_size):
            parameter_dict_list[_]["transPower"] = eval(inputFile.readline())


    def client_fn(cid: str):
        # create a single client instance
        return clt.FlowerClient(cid, fed_dir, parameter_dict_list[int(cid)])


    # (optional) specify Ray config
    ray_init_args = {
        "include_dashboard": True,
        "log_to_driver": True
    }

    # Configure the strategy
    strategy = SFL(
        on_fit_config_fn=clt.fit_config,
        # centralised evaluation of global model
        evaluate_fn=clt.get_evaluate_fn(testset),
    )

    # Configure the client manager
    if args.mode == "PPO":
        client_manager = PPO_ClientManager()
    elif args.mode == "Random":
        client_manager = Random_ClientManager()
    else:
        client_manager = SimpleClientManager()

    # start simulation
    simulation = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_manager=client_manager,
        ray_init_args=ray_init_args,
    )

    print(simulation)

    with open("./output/losses_centralized.txt", mode='w') as outputFile:
        outputFile.write(str(simulation.losses_centralized))
    with open("./output/losses_distributed.txt", mode='w') as outputFile:
        outputFile.write(str(simulation.losses_distributed))
    with open("./output/metrics_centralized.txt", mode='w') as outputFile:
        outputFile.write(str(simulation.metrics_centralized))

    ###############################################################################
    # Check records of last round
    with open(
            "./output/fit_server/round_{}.txt".format(num_rounds),
            mode='r'
    ) as last_inputFile:
        clients_of_last_round = eval(last_inputFile.readline())["clients_selected"]

    for _ in range(pool_size):
        # If the client was not selected in the last round,
        # help it complete the records
        if _ not in clients_of_last_round:
            with open(
                    "./output/train_loss/client_{}.txt".format(_),
                    mode='a'
            ) as last_outputFile:
                last_outputFile.write("-1" + "\n")
            with open(
                    "./output/val_accu/client_{}.txt".format(_),
                    mode='a'
            ) as outputFile:
                outputFile.write("-1" + "\n")
            with open(
                    "./output/val_loss/client_{}.txt".format(_),
                    mode='a'
            ) as outputFile:
                outputFile.write("-1" + "\n")
    
    # Record involvements of last round
    with open("./output/involvement_history.txt", mode='r') as inputFile:
        fileLine = inputFile.readline()
        assert fileLine
        involvement_history = eval(fileLine)
    for _ in range(pool_size):
        if _ in clients_of_last_round:
            involvement_history[_] += 1
    with open("./output/involvement_history.txt", mode='w') as outputFile:
        outputFile.write(str(involvement_history))
    ###############################################################################
