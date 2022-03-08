import numpy as np
import torch


def summarize_model(model: torch.nn.Module):
    """
    Generates a string, summarizing a pytorch model for an easy overview over sub-modules and parameter count

    :param model: The pytorch model (subclass of torch.nn.Module)
    :return: A string with one row for each submodule, with their names and parameter count
    """

    layers = [(name if len(name) > 0 else 'TOTAL', str(module.__class__.__name__), sum(np.prod(p.shape) for p in module.parameters())) for name, module in model.named_modules()]
    layers.append(layers[0])
    del layers[0]

    columns = [
        [" ", list(map(str, range(len(layers))))],
        ["Name", [layer[0] for layer in layers]],
        ["Type", [layer[1] for layer in layers]],
        ["Params", [layer[2] for layer in layers]],
    ]

    n_rows = len(columns[0][1])
    n_cols = 1 + len(columns)

    # Get formatting width of each column
    col_widths = []
    for c in columns:
        col_width = max(len(str(a)) for a in c[1]) if n_rows else 0
        col_width = max(col_width, len(c[0]))  # minimum length is header length
        col_widths.append(col_width)

    # Formatting
    s = "{:<{}}"
    total_width = sum(col_widths) + 3 * n_cols
    header = [s.format(c[0], l) for c, l in zip(columns, col_widths)]

    summary = " | ".join(header) + "\n" + "-" * total_width
    for i in range(n_rows):
        line = []
        for c, l in zip(columns, col_widths):
            line.append(s.format(str(c[1][i]), l))
        summary += "\n" + " | ".join(line)

    return summary


class FixedWithWarmupOptimizer:
    """
    Fixed learning rate after pre-defined number of warmup steps during which the learning rate linearly increases towards the final lr
    """

    def __init__(self, optimizer, final_lr, n_warmup_steps):
        self._optimizer = optimizer
        self.final_lr = final_lr
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self._optimizer.state_dict()

    def _update_learning_rate(self):
        self.n_steps += 1
        lr = self.final_lr * min(1., self.n_steps / self.n_warmup_steps)

        self.current_lr = lr

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
