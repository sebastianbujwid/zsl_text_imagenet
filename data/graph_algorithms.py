import collections
import logging
import tqdm
from typing import Mapping, Any, Set


def dfs_ancestors(node, parents, ancestors):
    if node in ancestors:
        return ancestors[node]

    node_ancestors = {node}
    for p in parents[node]:
        p_ancestors = dfs_ancestors(p, parents, ancestors)
        node_ancestors.update(p_ancestors)
        node_ancestors.add(p)

    ancestors[node] = node_ancestors
    return node_ancestors


def compute_ancestors(all_nodes, parents):
    ancestors = collections.defaultdict(set)
    for node in all_nodes:
        dfs_ancestors(node, parents, ancestors)
    return ancestors


def reachable_category_ancestors_all(parents: Mapping[Any, Mapping], max_depth: int = 20):
    # Need to call 'dfs_transistive_closure' on all categories, but some are not keys in 'parents'
    # - need to extract them from the dict values
    all_categories = set(parents.keys())
    for x in parents.values():
        all_categories.update(x['categories'])

    logging.info(f'Found all the categories: {len(all_categories)}')

    nodes_not_fully_expanded = set()
    reachability = collections.defaultdict(set)
    for c in tqdm.tqdm(all_categories, unit='categories'):
        dfs_transistive_closure(c, c, edges=parents,
                                reachability=reachability,
                                nodes_not_fully_expanded=nodes_not_fully_expanded,
                                max_depth=max_depth,
                                depth=1)
    return reachability, nodes_not_fully_expanded


def reachable_category_ancestors_single_category(category, parents: Mapping[Any, Mapping],
                                                 max_depth: int = 20):
    nodes_not_fully_expanded = set()
    reachability = collections.defaultdict(set)
    dfs_transistive_closure(category, category, edges=parents,
                            reachability=reachability,
                            nodes_not_fully_expanded=nodes_not_fully_expanded,
                            max_depth=max_depth,
                            depth=1)
    fully_expanded = category not in nodes_not_fully_expanded
    return reachability[category], fully_expanded


def dfs_transistive_closure(s, v, edges, reachability: Mapping[Any, Set], nodes_not_fully_expanded: Set,
                            max_depth: int, depth: int):
    """
    Based on DFS: Transistive Closure:
    https://www.cs.princeton.edu/courses/archive/spr03/cs226/lectures/digraph.4up.pdf
    """
    reachability[s].add(v)

    if v not in edges:
        return

    if depth >= max_depth:
        reachability[s].update(edges[v]['categories'])
        nodes_not_fully_expanded.add(s)
        return

    for w in edges[v]['categories']:
        # NOTE: shouldn't be used when using max_depth
        # if w in reachability:
        #     reachability[s].update(reachability[w])
        #     continue

        if w not in reachability[s]:
            dfs_transistive_closure(s, w, edges, reachability, nodes_not_fully_expanded, max_depth, depth + 1)
