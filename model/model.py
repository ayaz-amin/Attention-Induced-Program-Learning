import torch
import torch.nn as nn

from type_model import TypeModel
from rendering_model import RenderingModel
from deformation_model import DeformationModel


class AttentionInduction(nn.Module):
    '''
    The full Attention Induced Program Learning model. The model
    consists of three sub-models:
        a) Type model
        b) Rendering model
        c) Deformation (token) model

    Parameters
    ----------

    max_k: int
        Maximum number of parts that can be present in an image at once
    num_parts: int
        Number of possible sub-parts that can be used to construct an image
    filter_size: (int, int)
        The shape of the convolution kernels for convolving over the sparse
        representation of an image
    image_shape: (int, int, int)
        The shape of the (generated) image  
    '''

    def __init__(self, max_k=5, num_parts=10, filter_size=(64, 64), image_shape=(1, 105, 105)):
        super(AttentionInduction, self).__init__()
        '''
        Attributes
        ----------

        height, width: int, int
            Height and width of the image
        type_model: nn.Module
            The type model generates, well, a type
        rendering_model: nn.Module
            The rendering model renders the final image
        token_model: nn.Module
            The token model generates slight deformations onto the final image
        '''

        height, width = image_shape[1], image_shape[2]
        self.type_model = TypeModel(max_k=max_k, num_parts=num_parts, image_shape=(height, width))
        self.rendering_model = RenderingModel(num_parts=num_parts, filter_size=filter_size, image_shape=(height, width))
        self.token_model = DeformationModel(image_shape=image_shape)

    def forward(self):
        '''
        Forward generation of an image type

        Returns
        -------

        token_image: 2D torch.tensor
            The final generated image
        '''
        phw_list = self.type_model()
        type_img = self.rendering_model(phw_list)
        token_img = self.token_model(type_img)
        return token_img
