from math import log2

## Helper functions for computing the entropy and information gain
def entropy_helper(p):
    assert 0 <= p <= 1
    return -p*log2(p) if p != 0 else 0

def entropy(n1, n2):
    # n1: number of positive elements of the set
    # n2: number of negative elements of the set
    assert n1 >= 0 and n2 >= 0
    if n1 + n2 == 0:
        return 0
    return entropy_helper(n1 / (n1 + n2)) + entropy_helper(n2 / (n1 + n2))

def cond_entropy(nl1, nl2, nr1, nr2):
    # nl1: number of positive elements of one part of the split
    # nl2: number of negative elements of one part of the split
    # nr1: number of positive elements of the other part of the split
    # nr2: number of negative elements of the other part of the split
    assert nl1 >= 0 and nl2 >= 0 and nr1 >= 0 and nr2 >= 0
    n_elems = nl1 + nr1 + nl2 + nr2
    return (nl1 + nl2) / n_elems * entropy(nl1, nl2) \
         + (nr1 + nr2) / n_elems * entropy(nr1, nr2)

def inform_gain(nl1, nl2, nr1, nr2):
    # nl1: number of positive elements of one part of the split
    # nl2: number of negative elements of one part of the split
    # nr1: number of positive elements of the other part of the split
    # nr2: number of negative elements of the other part of the split
    assert nl1 >= 0 and nl2 >= 0 and nr1 >= 0 and nr2 >= 0
    return entropy(nl1 + nr1, nl2 + nr2) \
         - cond_entropy(nl1, nl2, nr1, nr2)


## Helper functions for creating graphviz.Digraph object from a Decision tree.
def make_digraph(node, graph, class_names, nid=0):
    res = nid
    if node is None:
        return None
    if not node.get('left'):
        add_digraph_node(nid, node, graph, class_names)
        res = nid + 1
    else:
        assert node.get('right')
        nid1 = make_digraph(node['left'], graph, class_names, nid + 1)
        assert nid1 is not None
        nid2 = make_digraph(node['right'], graph, class_names, nid1)
        add_digraph_node(nid2, node, graph, class_names)
        curr_id, left_id, right_id = nid2, nid1 - 1, nid2 - 1
        graph.edge(f'{curr_id}', f'{left_id}')
        graph.edge(f'{curr_id}', f'{right_id}')
        res = nid2 + 1
    return res

def add_digraph_node(nid, node, graph, class_names):
    label = ""
    if node.get('attr'):
        label = f"{node['attr']} < {round(node['split_value'], 4)}"
        label += '\n'
    label += f"samples = {node['samples']}"
    label += '\n'
    cl_name = class_names[node['class']]
    label += f"class = {cl_name}"
    label += '\n'
    pos_cl_name = class_names[1]
    neg_cl_name = class_names[0]
    neg_cl_num = node[neg_cl_name]
    pos_cl_num = node[pos_cl_name]
    label += f"{neg_cl_name} = {neg_cl_num},  {pos_cl_name} = {pos_cl_num}"
    # interpolate between red and green color RGB codes to color the more
    # "positive" nodes more redish, and the more "negative" nodes greenish.
    pos_cl_ratio = pos_cl_num / (pos_cl_num + neg_cl_num)
    node_color_rgb = (int(255*pos_cl_ratio), int(255*(1 - pos_cl_ratio)), 0)
    rgb2hex = lambda c: "#{:02x}{:02x}{:02x}".format(*c)
    graph.node(str(nid), label, style='filled', fillcolor=str(rgb2hex(node_color_rgb)))