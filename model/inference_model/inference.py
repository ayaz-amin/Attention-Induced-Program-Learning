import numpy as np
import networkx as nx
from numpy.random import rand, randint

from scipy.ndimage.morphology import grey_dilation as dilate_2d
from .preproc import Preproc


def infer(img, model_factors,
               pool_shape=(25, 25), num_candidates=5):
    """
    Main function for testing on one image.

    Parameters
    ----------
    img : 2D numpy.ndarray
        The testing image.
    model_factors : [(numpy.ndarray, networkx.Graph)]
        [(frcs, graphs)], output of train_image in learning.py.
    pool_shape : (int, int)
        Vertical and horizontal pool shapes.
    num_candidates : int
        Number of top candidates for backward-pass inference.

    Returns
    -------
    top_candidates: numpy.ndarray
        The top candidates for further backward-pass inference
    """
    # Get bottom-up messages from the pre-processing layer
    preproc_layer = Preproc(cross_channel_pooling=True)
    bu_msg = preproc_layer.fwd_infer(img)

    # Forward pass inference
    fp_scores = np.zeros(len(model_factors[0]))
    for i, (frcs, graph) in enumerate(model_factors):
        fp_scores[i] = forward_pass(frcs,
                                    bu_msg,
                                    graph,
                                    pool_shape)
    top_candidates = np.argsort(fp_scores)[-num_candidates:]

    return top_candidates


def forward_pass(frcs, bu_msg, graph, pool_shape):
    """
    Forward pass inference using a tree-approximation (cf. Sec S4.2).

    Parameters
    ----------
    frcs : numpy.ndarray of numpy.int
        Nx3 array of (feature idx, row, column), where each row represents a
        single pool center.
    bu_msg : 3D numpy.ndarray of float
        The bottom-up messages from the preprocessing layer.
        Shape is (num_feats, rows, cols)
    graph : networkx.Graph
        An undirected graph whose edges describe the pairwise constraints between
        the pool centers.
        The tightness of the constraint is in the 'perturb_radius' edge attribute.
    pool_shape : (int, int)
        Vertical and horizontal pool shapes.

    Returns
    -------
    fp_score : float
        Forward pass score.
    """
    height, width = bu_msg.shape[-2:]
    # Vertical and horizontal pool shapes
    vps, hps = pool_shape

    def _pool_slice(f, r, c):
        assert (r - vps // 2 >= 0 and r + vps - vps // 2 < height and
                c - hps // 2 >= 0 and c + hps - hps // 2 < width), \
            "Some pools are out of the image boundaries. "\
            "Consider increase image padding or reduce pool shapes."
        return np.s_[f,
                     r - vps // 2: r + vps - vps // 2,
                     c - hps // 2: c + hps - hps // 2]

    # Find a schedule to compute the max marginal for the most constrained tree
    tree_schedule = get_tree_schedule(frcs, graph)

    # If we're sending a message out from x to y, it means x has received all
    # incoming messages
    incoming_msgs = {}
    for source, target, perturb_radius in tree_schedule:
        msg_in = bu_msg[_pool_slice(*frcs[source])]
        if source in incoming_msgs:
            msg_in = msg_in + incoming_msgs[source]
            del incoming_msgs[source]
        msg_in = dilate_2d(msg_in, (2 * perturb_radius + 1, 2 * perturb_radius + 1))
        if target in incoming_msgs:
            incoming_msgs[target] += msg_in
        else:
            incoming_msgs[target] = msg_in
    fp_score = np.max(incoming_msgs[tree_schedule[-1, 1]] +
                      bu_msg[_pool_slice(*frcs[tree_schedule[-1, 1]])])
    return fp_score


def get_tree_schedule(frcs, graph):
    """
    Find the most constrained tree in the graph and returns which messages to compute
    it.  This is the minimum spanning tree of the perturb_radius edge attribute.

    See forward_pass for parameters.

    Returns
    -------
    tree_schedules : numpy.ndarray of numpy.int
        Describes how to compute the max marginal for the most constrained tree.
        Nx3 2D array of (source pool_idx, target pool_idx, perturb radius), where
        each row represents a single outgoing factor message computation.
    """
    min_tree = nx.minimum_spanning_tree(graph, 'perturb_radius')
    return np.array([(target, source, graph.edges[source, target]['perturb_radius'])
                     for source, target in nx.dfs_edges(min_tree)])[::-1]