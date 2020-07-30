import numpy as np

import torch
import torch.nn as nn


class RenderingModel(nn.Module):
    '''
    Rendering model for Attention Induced Program Learning

    Parameters 
    ----------

    num_parts: int
        Number of possible sub-parts that can be used to construct an image
    filter_size: (int, int)
        The shape of the convolution kernels for convolving over the sparse
        representation of an image
    image_shape: (int, int)
        The shape of the (generated) image 
    '''

    def __init__(self, num_parts=10, filter_size=(64, 64), image_shape=(105, 105)):
        super(RenderingModel, self).__init__()
        '''
        Attributes
        ----------

        height, width: int, int
            Height and width of final image
        filt_h, filt_w: int, int
            Filter height and width
        filters: 2D torch.tensor
            Filters for convolving over a sparse representation of an image
        '''

        self.height, self.width = image_shape[0], image_shape[1]
        self.filt_h, self.filt_w = filter_size[0], filter_size[1]
        self.filters = nn.Parameter(torch.randn((num_parts, self.filt_h, self.filt_w)))
        
    def forward(self, phw_list):
        '''
        Convolve over latent features
        Based on https://github.com/vicariousinc/science_rcn/blob/master/science_rcn/inference.py#L382 

        Parameters
        ----------
        phw_list: [(torch.tensor, torch.tensor, torch.tensor)]
            A list containing the feature index, the row, and the column of each part

        Returns
        -------
        canvas: 2D torch.tensor
              The final convolved image
        '''

        height, width = self.height, self.width
        filt_h, filt_w = self.filt_h, self.filt_w
        filt_o_h, filt_o_w = filt_h // 2, filt_w // 2

        from_r, to_r = (np.maximum(0, phw_list[:, 1] - filt_o_h),
                        np.minimum(height, phw_list[:, 1] - filt_o_h + filt_h))

        from_c, to_c = (np.maximum(0, phw_list[:, 2] - filt_o_w),
                        np.minimum(width, phw_list[:, 2] - filt_o_w + filt_w))

        from_fr, to_fr = (np.maximum(0, filt_o_h - phw_list[:, 1]),
                          np.minimum(filt_h, height - phw_list[:, 1] + filt_o_h))

        from_fc, to_fc = (np.maximum(0, filt_o_w - phw_list[:, 2]),
                          np.minimum(filt_w, width - phw_list[:, 2] + filt_o_w))

        canvas = torch.zeros((height, width))

        for i, (part_idx, row, column) in enumerate(phw_list):
            filts = self.filters[part_idx][from_fr[i]:to_fr[i], from_fc[i]:to_fc[i]]
            canvas[from_r[i]:to_r[i], from_c[i]:to_c[i]] += filts

        return canvas
