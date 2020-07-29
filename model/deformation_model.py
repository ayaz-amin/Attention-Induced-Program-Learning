import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformationModel(nn.Module):
    '''
    Deformation model for Attention Induced Program Learning

    Parameters
    ----------

    image_shape: (int, int, int)
        The shape of the generated image
    '''

    def __init__(self, image_shape=(1, 105, 105)):
        super(DeformationModel, self).__init__()
        '''
        Attributes
        ----------

        channels, height, width: int, int, int
            Channels, height and width of generated image
        localization_net: nn.Module
            Localization network for STN
        '''

        self.channels, self.height, self.width = image_shape[0], image_shape[1], image_shape[2]

        self.localization_net = nn.Sequential(
            nn.Conv2d(self.channels + 1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 6, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x, noise=None):
        '''
        Spatial Transformer Network (https://arxiv.org/abs/1506.02025)
        
        Parameters
        ----------

        x: 2D torch.tensor
            Input image
        
        Returns
        -------

        x_hat: 2D torch.tensor
            The deformed image
        '''

        if noise is None:
            noise = torch.randn(x.size())

        assert noise.size(0) == x.size(0), \
            "Heights are not the same"

        assert noise.size(1) == x.size(1), \
            "Widths are not the same"

        x = x.unsqueeze(0).unsqueeze(0)
        noise = noise.unsqueeze(0).unsqueeze(0)
        
        concat_x = torch.cat([x, noise], dim=1)

        theta = self.localization_net(concat_x)
        theta = theta.view(x.size(0), 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x_hat = F.grid_sample(x, grid, align_corners=True)

        return x_hat.squeeze(0).squeeze(0)
