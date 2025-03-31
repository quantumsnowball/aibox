from io import BytesIO

import matplotlib.pyplot as plt
import PIL.Image
import torch
import torch.nn as nn
from torchviz import make_dot


def show_backward_graph(output: torch.Tensor,
                        net: nn.Module | None = None,
                        figsize: tuple[float, float] = (12, 9),
                        dpi: str = '600',
                        **make_dot_kwargs) -> None:
    params = dict(net.named_parameters()) if net else None
    dot = make_dot(output, params=params, **make_dot_kwargs)
    dot.attr(dpi=dpi)
    graph_bytes = BytesIO(dot.pipe(format='png'))
    image = PIL.Image.open(graph_bytes)
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
