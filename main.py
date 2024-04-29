import argparse
import os
import shutil
from pathlib import Path

import flwr as fl
import torchvision

from dataset_utils import cifar10Transformation
from strategy import *
import client as clt

parser = argparse.ArgumentParser(description="SFL Device Selection PPO Training")
parser.add_argument("--num_client_cpus", type=int, default=6)
parser.add_argument("--num_client_gpus", type=int, default=1)
parser.add_argument("--mode", type=str, default="Full")

# Start simulation (a _default server_ will be created)
if __name__ == "__main__":
    # parse input arguments
    args = parser.parse_args()

    fed_dir = "./data/cifar-10-batches-py/federated/"
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=cifar10Transformation()
    )

    # clear previous records
    path_to_init = ["train_loss_per_cli", "val_accu_per_cli", "val_loss_per_cli"]
    for _ in path_to_init:
        if Path("output/" + _ + "/").exists():
            shutil.rmtree("output/" + _ + "/")
        os.mkdir("output/" + _ + "/")
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
    strategy = FL(
        on_fit_config_fn=clt.fit_config,
        # centralised evaluation of global model
        evaluate_fn=clt.get_evaluate_fn(testset),
    )

    # Configure the client manager
    if args.mode == "Full":
        client_manager = Full_ClientManager()
    else:
        client_manager = SimpleClientManager()
        raise NameError("Invalid mode")

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
