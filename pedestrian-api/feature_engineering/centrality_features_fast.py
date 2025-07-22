import random
import networkx as nx

def compute_centrality_fast(G, edges_gdf, k=500):
    """
    Sampled betweenness and closeness centrality:
    - G: NetworkX graph
    - edges_gdf: GeoDataFrame with an 'osmid' column
    - k: number of source nodes to sample
    Returns two dicts: {osmid: betweenness}, {osmid: closeness}
    """
    # 1) sample k nodes
    nodes = list(G.nodes())
    if k < len(nodes):
        sample = random.sample(nodes, k)
    else:
        sample = nodes

    # 2) compute betweenness on sample
    betweenness = nx.betweenness_centrality_subset(
        G, sources=sample, targets=nodes, normalized=True, weight="length"
    )

    # 3) compute closeness on sample and average
    closeness = {}
    for u in sample:
        c = nx.closeness_centrality(G, u, distance="length")
        for v in edges_gdf["osmid"]:
            # map each edge’s osmid to its centrality; fallback to 0
            closeness[v] = closeness.get(v, 0) + c
    # normalize closeness by number of samples
    for v in closeness:
        closeness[v] /= len(sample)

    return betweenness, closeness 