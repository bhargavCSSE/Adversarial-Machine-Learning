How to demo this software?
1. Recommended python interpreter - python 3.6
2. Make sure you have all the packages mentioned in the requirement.txt installed in your python environment.
3. Run NeuroPLC.py for evolving architecture for NeuroPLC dataset. (This is faster to execute)
    a. Change runCount (line 318) to any number of runs you want to perform.
    b. You can also change the population size and the mutation amount.
    c. Chromosome length should be the maximum amount of hidden layers you would want in your architecture.
4. Run MNIST.py for evolving architecture for MNIST dataset. (This takes a while to finish executing)
5. Both programs create the final result as Results_{dataset}.csv file.

