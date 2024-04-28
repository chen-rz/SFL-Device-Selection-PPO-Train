import random
import argparse

from constants import *
from dataset_utils import get_cifar_10, do_fl_partitioning

parser = argparse.ArgumentParser(description="Partition Dataset")
parser.add_argument(
    "--alpha", type=int, default=1,
    help='''
        Use a large `alpha` (e.g. 1000) to make it IID; a small value (e.g. 1) will make it non-IID. 
        This will create a new directory called "federated" in the directory where CIFAR-10 lives. 
        Inside it, there will be N=pool_size subdirectories each with its own train/set split.
        '''
)
args = parser.parse_args()

# Download CIFAR-10 dataset
train_path, testset = get_cifar_10()

# Partition dataset
fed_dir = do_fl_partitioning(
    train_path, pool_size=pool_size, alpha=args.alpha, num_classes=10, val_ratio=0.1
)

# Record dataset sizes (number of data instances)
with open("./parameters/dataSize.txt", mode='w') as outputFile:
    outputFile.write("")
with open("./output/label_distribution_histograms.txt", mode='r') as inputFile, \
    open("./parameters/dataSize.txt", mode='a') as outputFile:
    for n in range(pool_size):
        hist_line = inputFile.readline()
        assert hist_line
        outputFile.write(str(sum(eval(hist_line))) + "\n")
print("Dataset initialization completed")

# Define CPU/GPU computational capabilities (FLOPs)
with open("./parameters/computation.txt", mode='w') as outputFile:
    for n in range(pool_size):
        outputFile.write(str(random.gauss(mu=6e9, sigma=1.2e9)) + "\n")
        #(str(random.uniform(2e9, 10e9)) + "\n")
print("CPU/GPU computational capabilites initialization completed")

# Define transmission power
with open("./parameters/transPower.txt", mode='w') as outputFile:
    for n in range(pool_size):
        outputFile.write(str(random.uniform(1e-3, 5e-3)) + "\n")
print("Transmission power initialization completed")
