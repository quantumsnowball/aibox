from io import BytesIO

import matplotlib.pyplot as plt
import PIL.Image
import torch
import torch.nn as nn
from torch.autograd.graph import Node
from torchviz import make_dot


def plot_backward_graph(output: torch.Tensor,
                        net: nn.Module | None = None,
                        figsize: tuple[float, float] = (12, 9),
                        dpi: str = '600',
                        **make_dot_kwargs) -> None:
    # auto get named parameters from net
    params = dict(net.named_parameters()) if net else None

    # create the Digraph
    dot = make_dot(output, params=params, **make_dot_kwargs)
    dot.attr(dpi=dpi)

    # create bytes io for the graph
    graph_bytes = BytesIO(dot.pipe(format='png'))
    image = PIL.Image.open(graph_bytes)

    # show the graph using matplotlib
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def print_backward_graph(output: torch.Tensor,
                         model: torch.nn.Module | None = None):
    # if model is suppied, derive the leaf node names
    leaf_nodes = list(dict(model.named_parameters()).keys()) if model else None

    def print_backward_fn(fn: Node, i: int = 0):
        # we arrive at a grad_fn, print its name
        print(f'{i*'|   '}{fn.name()}', end='')

        # point to one or more next functions, should be an intermediate node
        if len(fn.next_functions) > 0:
            # iter the list and handle with recursion
            for next_fn, _ in fn.next_functions:
                if not next_fn:
                    continue
                print('\n', end='')
                print_backward_fn(next_fn, i=i+1)
        # no next functions, should be follow by a leaf node
        else:
            # print its name from if available
            if leaf_nodes:
                print(f' <- {leaf_nodes.pop(0)}', end='')
            else:
                print(' <- (Param)', end='')

        if i == 0:
            print('\n', end='')

    if fn := output.grad_fn:
        print_backward_fn(fn)
