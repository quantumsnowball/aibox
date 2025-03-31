from io import BytesIO

import matplotlib.pyplot as plt
import PIL.Image
import torch
from torchviz import make_dot


def show_backward_graph(output: torch.Tensor,
                        *make_dot_args,
                        figsize=(12, 9),
                        dpi='600',
                        **make_dot_kwargs) -> None:
    dot = make_dot(output, *make_dot_args, **make_dot_kwargs)
    dot.attr(dpi=dpi)
    graph_bytes = BytesIO(dot.pipe(format='png'))
    image = PIL.Image.open(graph_bytes)
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
