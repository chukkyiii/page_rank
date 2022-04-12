import sys
import os
import time
import argparse
from progress import Progress
from random import randint, choice
from numpy import sum

def load_graph(args):
    """Load graph from text file

    Parameters:
    args -- arguments named tuple

    Returns:
    A dict mapling a URL (str) to a list of target URLs (str)."""
    dict = {}
    last_node = None
    # Iterate through the file line by line
    for line in args.datafile:
        # And split each line into two URLs
        node, target = line.split()

        if node != last_node:
            dict[node] = [target]
            # Using Target as a an array 
        else:
            dict[node].append(target)
        last_node = node
    return dict

def print_stats(graph):
        """Print number of nodes and edges in the given graph"""
        no_node = len(graph)

        no_edges = 0
        # for i in graph.values():
        #     no_edges += len(i)
        
        no_edges = sum([len(i) for i in graph.values()])
        # Taking the sum of each length in each value array. 
        print(no_node, no_edges)


def stochastic_page_rank(graph, args):
    """Stochastic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    args -- arguments named tuple

    Returns:
    A dict that assigns each page its hit frequency

    This function estimates the Page Rank by counting how frequently
    a random walk that starts on a random node will after n_steps end
    on each node of the given graph.
    """
    # raise RuntimeError("This function is not implemented yet.")

    hit_count = {node: 0 for node in graph}
    # each node corresponds to the value 0 
    for _ in range(args.repeats):
        current_node = choice(list(graph))
        # curent_node chooses a random node. 
        for _ in range(args.steps):
            current_node = graph[current_node][randint(0, len(graph[current_node]) - 1)]
            # Chooses a random edge. 
            # graph -> {node1: [url1_0, url1_1, url1_2, ... ], node2: [url2_0, url2_1, url2_2, ...] ...}
            # graph[current_node] -> [url1, url2, url3, ...]
            # graph[current_node][x] -> [urlx]
        hit_count[current_node] += 1 / args.repeats

    return hit_count


def distribution_page_rank(graph, args):
    """Probabilistic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    args -- arguments named tuple

    Returns:
    A dict that assigns each page its probability to be reached

    This function estimates the Page Rank by iteratively calculating
    the probability that a random walker is currently on any node.
    """
    node_prob = {node: 1 / len(graph) for node in graph}
    # Each node has the value 1 / len(graph) (1/555) 
    for _ in range(args.steps):
        next_prob = {node: 0 for node in graph}
        # each node is reset to 0
        # Attempt 2:
        # >>> p = node_prob[choice(list(graph))]
        # Attempt 1:
        # >>> for node in graph: 
        # >>>   p = node_prob[node]
        p = 1 / len(graph)
        # no matter what p = 1 / len(graph) no need for for loop. 
        for targets in graph.values():
            for target in targets:
                next_prob[target] += p
        # next_prob = {target: next_prob[target] + p for targets in graph.values() for target in targets} 
        # ^^^ This does not give the right output, and its more ineffiecent ^^^
        # ...nested for loop :(
    node_prob = next_prob
    return node_prob


    # raise RuntimeError("This function is not implemented yet.")


parser = argparse.ArgumentParser(description="Estimates page ranks from link information")
parser.add_argument('datafile', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                    help="Textfile of links among web pages as URL tuples")
parser.add_argument('-m', '--method', choices=('stochastic', 'distribution'), default='stochastic',
                    help="selected page rank algorithm")
parser.add_argument('-r', '--repeats', type=int, default=1_000_000, help="number of repetitions")
parser.add_argument('-s', '--steps', type=int, default=100, help="number of steps a walker takes")
parser.add_argument('-n', '--number', type=int, default=20, help="number of results shown")


if __name__ == '__main__':
    args = parser.parse_args()
    algorithm = distribution_page_rank if args.method == 'distribution' else stochastic_page_rank

    graph = load_graph(args)

    print_stats(graph)

    start = time.time()
    ranking = algorithm(graph, args)
    stop = time.time()
    time = stop - start

    top = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    sys.stderr.write(f"Top {args.number} pages:\n")
    print('\n'.join(f'{100*v:.2f}\t{k}' for k,v in top[:args.number]))
    sys.stderr.write(f"Calculation took {time:.2f} seconds.\n")



'''
Your job is to implement the functions such that read_graph returns
some python object that contains the graph data. stochastic_page_rank
should implement the first method explained above to estimate PageRanks
via random walkers, whereas distribution_page_rank should implement the
second method to estimate PageRanks via probability distributions.
The functions that estimate PageRanks have to follow exactly the
behaviour specified by the above pseudo code definitions.
'''

'''
file should have the following format:
start_url end_url
...
'''
