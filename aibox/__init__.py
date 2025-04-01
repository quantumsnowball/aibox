from io import BytesIO

import matplotlib.pyplot as plt
import PIL.Image
import torch
import torch.nn as nn
from torch.autograd.graph import Node
from torchviz import make_dot


def show_backward_graph(output: torch.Tensor,
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


def print_backward_graph(output: torch.Tensor):
    def print_backward_fn(fn: Node | None, i: int = 0):
        if not fn:
            return
        print(f'{i*'|   '}{fn.name()}')
        for next_fn, _ in fn.next_functions:
            print_backward_fn(next_fn, i=i+1)

    print_backward_fn(output.grad_fn)
