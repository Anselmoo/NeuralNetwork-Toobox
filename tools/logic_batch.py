"""Analytic 
"""
import multiprocessing as mp
import logic_generator as lg
from tqdm import tqdm


def get_totalCores():
    """Get the total numbers of cores."""
    print("Number of Cores:", mp.cpu_count())


def generator_func(items=1):
    """Reference function for multiprocessing.

    Parameters
    ----------
    items : int
        Number of items in the logic table.
    """
    lg.LogicGenerator(items).generator()


def batch_run(n_items):
    """Run a batch of logic generators.

    Parameters
    ----------
    n_items : list
        Number of items in the truth table.
    """

    procs = []
    # instantiating without any argument
    proc = mp.Process(target=generator_func)  # instantiating without any argument
    procs.append(proc)
    proc.start()
    # instantiating process with arguments
    for c_item in n_items:
        # print(name)
        proc = mp.Process(target=generator_func, args=(c_item,))
        procs.append(proc)
        proc.start()
    # complete the processes
    for proc in tqdm(procs):
        proc.join()
if __name__ == "__main__":
    get_totalCores()
    batch_run(n_items=[2, 3, 4, 5, 6, 7, 8, 9, 10])
