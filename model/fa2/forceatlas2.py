import random
import time

import numpy
import scipy
from tqdm import tqdm

from . import fa2util


class Timer:
    def __init__(self, name="Timer"):
        self.name = name
        self.start_time = 0.0
        self.total_time = 0.0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.total_time += (time.time() - self.start_time)

    def display(self):
        print(self.name, " took ", "%.2f" % self.total_time, " seconds")


class ForceAtlas2:
    def __init__(self,
                 # Behavior alternatives
                 outboundAttractionDistribution=False,  # Dissuade hubs
                 linLogMode=False,  # NOT IMPLEMENTED
                 adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                 edgeWeightInfluence=1.0,

                 # Performance
                 jitterTolerance=1.0,  # Tolerance
                 barnesHutOptimize=True,
                 barnesHutTheta=1.2,
                 multiThreaded=False,  # NOT IMPLEMENTED

                 # Tuning
                 scalingRatio=2.0,
                 strongGravityMode=False,
                 gravity=1.0,

                 # Log
                 verbose=True):
        assert linLogMode == adjustSizes == multiThreaded == False, "You selected a feature that has not been implemented yet..."
        self.outboundAttractionDistribution = outboundAttractionDistribution
        self.linLogMode = linLogMode
        self.adjustSizes = adjustSizes
        self.edgeWeightInfluence = edgeWeightInfluence
        self.jitterTolerance = jitterTolerance
        self.barnesHutOptimize = barnesHutOptimize
        self.barnesHutTheta = barnesHutTheta
        self.scalingRatio = scalingRatio
        self.strongGravityMode = strongGravityMode
        self.gravity = gravity
        self.verbose = verbose

    def init(self,
             G,  # a graph in 2D numpy ndarray format (or) scipy sparse matrix format
             pos=None  # Array of initial positions
             ):
        isSparse = False
        if isinstance(G, numpy.ndarray):
            # Check our assumptions
            assert G.shape == (G.shape[0], G.shape[0]), "G is not 2D square"
            assert numpy.all(G.T == G), "G is not symmetric.  Currently only undirected graphs are supported"
            assert isinstance(pos, numpy.ndarray) or (pos is None), "Invalid node positions"
        elif scipy.sparse.issparse(G):
            # Check our assumptions
            assert G.shape == (G.shape[0], G.shape[0]), "G is not 2D square"
            assert isinstance(pos, numpy.ndarray) or (pos is None), "Invalid node positions"
            G = G.tolil()
            isSparse = True
        else:
            assert False, "G is not numpy ndarray or scipy sparse matrix"

        # Put nodes into a data structure we can understand
        nodes = []
        for i in range(0, G.shape[0]):
            n = fa2util.Node()
            if isSparse:
                n.mass = 1 + len(G.rows[i])
            else:
                n.mass = 1 + numpy.count_nonzero(G[i])
            n.old_dx = 0
            n.old_dy = 0
            n.dx = 0
            n.dy = 0
            if pos is None:
                n.x = random.random()
                n.y = random.random()
            else:
                n.x = pos[i][0]
                n.y = pos[i][1]
            nodes.append(n)

        # Put edges into a data structure we can understand
        edges = []
        es = numpy.asarray(G.nonzero()).T
        for e in es:  # Iterate through edges
            if e[1] <= e[0]: continue  # Avoid duplicate edges
            edge = fa2util.Edge()
            edge.node1 = e[0]  # The index of the first node in `nodes`
            edge.node2 = e[1]  # The index of the second node in `nodes`
            edge.weight = G[tuple(e)]
            edges.append(edge)

        return nodes, edges

    def forceatlas2(self,
                    G,  # a graph in 2D numpy ndarray format (or) scipy sparse matrix format
                    pos=None,  # Array of initial positions
                    iterations=100  # Number of times to iterate the main loop
                    ):
        speed = 1.0
        speedEfficiency = 1.0
        nodes, edges = self.init(G, pos)
        outboundAttCompensation = 1.0
        if self.outboundAttractionDistribution:
            outboundAttCompensation = numpy.mean([n.mass for n in nodes])

        barneshut_timer = Timer(name="BarnesHut Approximation")
        repulsion_timer = Timer(name="Repulsion forces")
        gravity_timer = Timer(name="Gravitational forces")
        attraction_timer = Timer(name="Attraction forces")
        applyforces_timer = Timer(name="AdjustSpeedAndApplyForces step")

        # Each iteration of this loop represents a call to goAlgo().
        niters = range(iterations)
        if self.verbose:
            niters = tqdm(niters)
        for _i in niters:
            for n in nodes:
                n.old_dx = n.dx
                n.old_dy = n.dy
                n.dx = 0
                n.dy = 0

            # Barnes Hut optimization
            if self.barnesHutOptimize:
                barneshut_timer.start()
                rootRegion = fa2util.Region(nodes)
                rootRegion.buildSubRegions()
                barneshut_timer.stop()

            # Charge repulsion forces
            repulsion_timer.start()
            # parallelization should be implemented here
            if self.barnesHutOptimize:
                rootRegion.applyForceOnNodes(nodes, self.barnesHutTheta, self.scalingRatio)
            else:
                fa2util.apply_repulsion(nodes, self.scalingRatio)
            repulsion_timer.stop()

            # Gravitational forces
            gravity_timer.start()
            fa2util.apply_gravity(nodes, self.gravity, scalingRatio=self.scalingRatio, useStrongGravity=self.strongGravityMode)
            gravity_timer.stop()

            # If other forms of attraction were implemented they would be selected here.
            attraction_timer.start()
            fa2util.apply_attraction(nodes, edges, self.outboundAttractionDistribution, outboundAttCompensation,
                                     self.edgeWeightInfluence)
            attraction_timer.stop()

            # Adjust speeds and apply forces
            applyforces_timer.start()
            values = fa2util.adjustSpeedAndApplyForces(nodes, speed, speedEfficiency, self.jitterTolerance)
            speed = values['speed']
            speedEfficiency = values['speedEfficiency']
            applyforces_timer.stop()

        if self.verbose:
            if self.barnesHutOptimize:
                barneshut_timer.display()
            repulsion_timer.display()
            gravity_timer.display()
            attraction_timer.display()
            applyforces_timer.display()
        return [(n.x, n.y) for n in nodes]

    def forceatlas2_networkx_layout(self, G, pos=None, iterations=100, weight_attr=None):
        import networkx
        try:
            import cynetworkx
        except ImportError:
            cynetworkx = None

        assert (
            isinstance(G, networkx.classes.graph.Graph)
            or (cynetworkx and isinstance(G, cynetworkx.classes.graph.Graph))
        ), "Not a networkx graph"
        assert isinstance(pos, dict) or (pos is None), "pos must be specified as a dictionary, as in networkx"
        M = networkx.to_scipy_sparse_matrix(G, dtype='f', format='lil', weight=weight_attr)
        if pos is None:
            l = self.forceatlas2(M, pos=None, iterations=iterations)
        else:
            poslist = numpy.asarray([pos[i] for i in G.nodes()])
            l = self.forceatlas2(M, pos=poslist, iterations=iterations)
        return dict(zip(G.nodes(), l))

    # A layout for igraph.
    #
    # This function returns an igraph layout
    def forceatlas2_igraph_layout(self, G, pos=None, iterations=100, weight_attr=None):

        from scipy.sparse import csr_matrix
        import igraph

        def to_sparse(graph, weight_attr=None):
            edges = graph.get_edgelist()
            if weight_attr is None:
                weights = [1] * len(edges)
            else:
                weights = graph.es[weight_attr]

            if not graph.is_directed():
                edges.extend([(v, u) for u, v in edges])
                weights.extend(weights)

            return csr_matrix((weights, zip(*edges)))

        assert isinstance(G, igraph.Graph), "Not a igraph graph"
        assert isinstance(pos, (list, numpy.ndarray)) or (pos is None), "pos must be a list or numpy array"

        if isinstance(pos, list):
            pos = numpy.array(pos)

        adj = to_sparse(G, weight_attr)
        coords = self.forceatlas2(adj, pos=pos, iterations=iterations)

        return igraph.layout.Layout(coords, 2)
