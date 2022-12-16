from preprocessing import term_matrix
from fa2 import ForceAtlas2
from networkx.readwrite import json_graph
import networkx as nx
import matplotlib.pyplot as plt

def get_network(filepath : str, openwith : str):
    df = term_matrix(filepath = filepath, openwith = "url")
    graph = nx.from_numpy_matrix(df.values)
    G = graph
    
    forceatlas2 = ForceAtlas2(outboundAttractionDistribution=True,  # Dissuade hubs
                          linLogMode=False,  # NOT IMPLEMENTED
                          adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                          edgeWeightInfluence=1.0,
                          
                          jitterTolerance=5.0,  # Tolerance
                          barnesHutOptimize=True,
                          barnesHutTheta=1.2,
                          multiThreaded=False,  # NOT IMPLEMENTED

                          # Tuning
                          scalingRatio=2.0,
                          strongGravityMode=False,
                          gravity=1.0,

                          # Log
                          verbose=True)
    positions = forceatlas2.forceatlas2_networkx_layout(G, 
                                                        pos = None, 
                                                        iterations=2000)
    # largest connected component
    components = nx.connected_components(G)
    largest_component = max(components, key=len)
    H = G.subgraph(largest_component)
    
    
    # Networkx layout styles 
    layouts = {
        'spring': nx.spring_layout(G), 
        'spectral':nx.spectral_layout(G), 
        'shell':nx.shell_layout(G), 
        'circular':nx.circular_layout(G),
        'kamada_kawai':nx.kamada_kawai_layout(G), 
        'random':nx.random_layout(G)
        }

    # compute centrality
    centrality = nx.betweenness_centrality(H, k=10, 
                                           endpoints=True)

    # compute community structure
    lpc = nx.community.label_propagation_communities(H)
    community_index = {n: i for i, com in enumerate(lpc) for n in com}

    #### draw graph ####
    fig, ax = plt.subplots(figsize=(20, 15))
    pos = nx.spring_layout(H, k=0.15, seed=4572321)
    node_color = [community_index[n] for n in H]
    node_size = [v * 20000 for v in centrality.values()]


    nx.draw_networkx(
        H,
        pos=pos,
        with_labels=False,
        node_color=node_color,
        node_size=node_size,
        edge_color="gainsboro",
        alpha=0.4,
    )

    # Title/legend
    font = {"color": "k", "fontweight": "bold", "fontsize": 20}
    ax.set_title("test!", font)
    font["color"] = "r" # Change font color for legend

    ax.text(
        0.80,
        0.10,
        "node color = community structure",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )
    ax.text(
        0.80,
        0.06,
        "node size = betweeness centrality",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )

    # Resize figure for label readibility
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    plt.savefig("../figure/savefig_default.png")

    json_graph.node_link_data(H)


get_network(filepath = "https://raw.githubusercontent.com/e9t/nsmc/master/synopses.json", 
            openwith = 'url')