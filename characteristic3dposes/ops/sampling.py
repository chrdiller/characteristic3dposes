import torch
from torch.nn import functional


def scaling_fun(heatmap: torch.Tensor, strategy='softmax', temperature=1.) -> torch.Tensor:
    """
    Scale a given heatmap with the given strategy and temperature

    :param heatmap: Tensor of any dimensionality to be scaled
    :param strategy: softmax (applies a simple softmax with temperature) or simple (divides by temperature, subtracts min, and divides by sum)
    :param temperature: Temperature to be used
    :return: Same tensor shape as input, with values scaled according to strategy and temperature
    """

    if strategy == 'softmax':
        return functional.softmax(heatmap.flatten() / temperature, dim=-1).view_as(heatmap)
    elif strategy == 'simple':
        temperature_scaled = heatmap / temperature
        shifted = temperature_scaled - temperature_scaled.min()
        scaled = (torch.ones_like(shifted) / shifted.numel()) if shifted.sum() == 0 else (shifted / shifted.sum())
        return scaled
    else:
        raise ValueError


class JointSampler(torch.nn.Module):
    """Samples joint locations from given heatmaps and offset volumes"""

    def __init__(self, heatmap_resolution: int, heatmap_sigma: int, use_offsets=True):
        """
        Initializes the JointSampler: Construct box filter, set parameters

        :param heatmap_resolution: The spatial resolution of the heatmap, in voxels
        :param heatmap_sigma: The sigma value to be used in the Gaussian blur
        :param offsets: Whether to add predicted offsets to the sampled joint locations
        """

        super().__init__()
        self.box_filter = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=2*heatmap_sigma+1, stride=1, padding=heatmap_sigma, padding_mode='replicate').requires_grad_(False)
        self.box_filter.weight.fill_(1)
        self.box_filter.bias.fill_(0)
        self.heatmap_resolution = heatmap_resolution
        self.use_offsets = use_offsets

    def forward(self, heatmap: torch.Tensor, offsets: torch.Tensor, k: int, volume_center=None, temperature=0.025, nms=False) -> torch.Tensor:
        """
        Sample from given heatmap and add offsets (No batching, only single heatmap). Sample k samples. Volume will be centered at volume_center or at 0 if None.
        Before sampling, will apply a box filter with size 2*sigma+1 and scale the result with a softmax function

        :param heatmap: The 3D heatmap volume
        :param offsets: The 3D offset volume
        :param k: Number of samples to product
        :param volume_center: The center of the volume in world coordinates or None to center at 0
        :param temperature: The temperatur to be used for softmax
        :param nms: Whether to sample with nms
        :return: Tensor of size k x 3 with the samples in world coordinates
        """

        if nms:
            max_indices = torch.nonzero(heatmap.transpose(2, 1) == heatmap.max(), as_tuple=False)
            if k-len(max_indices) > 0:
                max_indices = torch.cat([max_indices, max_indices[-1:].repeat_interleave(k - len(max_indices), 0)])

            samples_gridspace = max_indices[::len(max_indices)//k][:k]
            samples_gridspace = samples_gridspace[:, (0, 2, 1)]
        else:
            internal_heatmap = self.box_filter(heatmap[None, None, :])[0, 0]
            internal_heatmap = functional.softmax(internal_heatmap.flatten() / temperature, dim=0).view(self.heatmap_resolution, self.heatmap_resolution, self.heatmap_resolution)
            samples_gridspace = torch.multinomial(internal_heatmap.flatten(), k, replacement=False)

            samples_gridspace = torch.stack([samples_gridspace // (self.heatmap_resolution ** 2), samples_gridspace // self.heatmap_resolution % self.heatmap_resolution, samples_gridspace % self.heatmap_resolution], 1).contiguous()
        samples = samples_gridspace / (self.heatmap_resolution / 2) - 1. + (volume_center if volume_center is not None else 0)

        if offsets is not None and self.use_offsets:
            samples = samples + offsets[:, samples_gridspace[:, 0], samples_gridspace[:, 1], samples_gridspace[:, 2]].T

        return samples

    def backward(self):
        raise NotImplementedError


if __name__ == '__main__':
    sampler = JointSampler(heatmap_resolution=16, heatmap_sigma=2, use_offsets=True)
    heatmap = torch.zeros(size=[16, 16, 16])
    heatmap[8, 8, 8] = 1
    heatmap = scaling_fun(heatmap, strategy='simple', temperature=0.01)
    offsets = torch.zeros(size=[3, 16, 16, 16])
    offsets[:, 8, 8, 8] = torch.tensor([1, 0, -1])

    samples = sampler(heatmap, offsets, k=10)
    print(samples)
