import numpy as np

import torch
import torch.nn as nn

from torch.distributions import Categorical


class TypeModel(nn.Module):
    '''
    Type model (uncoditional) for Attention Induced Program Learning

    Parameters
    ----------

    max_k: int
        Maximum number of parts that can be present in an image at once
    num_parts: int
        Number of possible sub-parts that can be used to construct an image
    image_shape: (int, int)
        The shape of the (generated) image  
    '''

    def __init__(self, max_k=5, num_parts=10, image_shape=(105, 105)):
        super(TypeModel, self).__init__()

        '''
        Attributes
        ----------

        k: torch.tensor
            Tensor specifying the probability of selecting part count
        P: 2D torch.tensor
            Tensor (2D) specifying the probability of selecting a part at part-count k
        cd_h, cd_w: int, int
            Height and width boundaries for restricting the sampling of coordinates
        height_matrix: 2D torch.tensor
            Tensor (2D) specifying the probability of selecting height index (column) at part-count k
        width_matrix: 2D torch.tensor
            Tensor (2D) specifying the probability of selecting width index (row) at part-count k
        '''

        self.k = nn.Parameter(torch.ones((max_k)) / 1.0 * max_k)
        self.P = nn.Parameter(torch.ones((max_k, num_parts)) / 1.0 * num_parts)

        cd_h, cd_w = image_shape[0], image_shape[1]
        self.height_matrix = nn.Parameter(torch.ones((max_k, cd_h)) / 1.0 * cd_h)
        self.width_matrix = nn.Parameter(torch.ones((max_k, cd_w)) / 1.0 * cd_w)

    def sample_k(self):
        '''
        Sample number of sub-parts

        Returns
        -------
        k: torch.tensor
            Part-count
        '''

        dist = Categorical(self.k)
        return dist.sample()
    
    def sample_part(self, k_i):
        '''
        Sample sub-part at part-count k_i

        Parameters
        ----------
        k_i: torch.tensor
            Ith part-count
        
        Returns
        -------
        part_i: torch.tensor
            Ith sub-part index
        '''

        P = self.P[k_i]
        dist = Categorical(P)
        return dist.sample()
    
    def sample_coordinate(self, k_i):
        '''
        Sample coordinates of sub-parts

        Parameters
        ----------
        k_i: torch.tensor
            Ith part-count
        
        Returns
        -------
        h_i, w_i: torch.tensor, torch.tensor
            Ith height and width
        '''

        cd_h = self.height_matrix[k_i]
        cd_w = self.width_matrix[k_i]

        dist_h = Categorical(cd_h)
        dist_w = Categorical(cd_w)

        return dist_h.sample(), dist_w.sample()

    def forward(self):
        '''
        Generate image

        Returns
        -------
        phw_list: [(torch.tensor, torch.tensor, torch.tensor)]
            A list containing the feature index, the row, and the column of each part
            Needed for convolving over the final image
        '''

        k = self.sample_k()
        phw_list = []

        for i in range(k):
            part_i = self.sample_part(i)
            h_i, w_i = self.sample_coordinate(i)
            phw_list.append((part_i, h_i, w_i))

        if len(phw_list) == 0:
            self.forward()

        return np.array(phw_list)


class ConditionalTypeModel(nn.Module):
    '''
    Type model (conditional) for Attention Induced Program Learning

    Parameters
    ----------

    num_classes: int
        The number of classes that can be sampled from the model
    max_k: int
        Maximum number of parts that can be present in an image at once
    num_parts: int
        Number of possible sub-parts that can be used to construct an image
    image_shape: (int, int)
        The shape of the (generated) image  
    '''

    def __init__(self, num_classes, max_k=5, num_parts=10, image_shape=(105, 105)):
        super(ConditionalTypeModel, self).__init__()

        '''
        Attributes
        ----------

        k: 2D torch.tensor
            Tensor (2D) specifying the probability of selecting part count
        P: 3D torch.tensor
            Tensor (3D) specifying the probability of selecting a part at part-count k
        cd_h, cd_w: int, int
            Height and width boundaries for restricting the sampling of coordinates
        height_matrix: 3D torch.tensor
            Tensor (3D) specifying the probability of selecting height index (column) at part-count k
        width_matrix: 3D torch.tensor
            Tensor (3D) specifying the probability of selecting width index (row) at part-count k
        '''

        self.num_classes = num_classes

        self.k = nn.Parameter(torch.ones((num_classes, max_k)) / 1.0 * max_k)
        self.P = nn.Parameter(torch.ones((num_classes, max_k, num_parts)) / 1.0 * num_parts)

        cd_h, cd_w = image_shape[0], image_shape[1]
        self.height_matrix = nn.Parameter(torch.ones((num_classes, max_k, cd_h)) / 1.0 * cd_h)
        self.width_matrix = nn.Parameter(torch.ones((num_classes, max_k, cd_w)) / 1.0 * cd_w)

    def sample_k(self, class_idx):
        '''
        Sample number of sub-parts

        Parameters
        ----------

        class_idx: int
            The class index

        Returns
        -------
        k: torch.tensor
            Part-count
        '''

        k_class = self.k[class_idx]
        dist = Categorical(k_class)
        k = dist.sample()
        if k == 0:
            k = 1
        return k
    
    def sample_part(self, class_idx, k_i):
        '''
        Sample sub-part at part-count k_i

        Parameters
        ----------
        class_idx: int
            The class index

        k_i: torch.tensor
            Ith part-count
        
        Returns
        -------
        part_i: torch.tensor
            Ith sub-part index
        '''

        P_class = self.P[class_idx]
        P_dist = P_class[k_i]
        dist = Categorical(P_dist)
        return dist.sample()
    
    def sample_coordinate(self, class_idx, k_i):
        '''
        Sample coordinates of sub-parts

        Parameters
        ----------
        class_idx: int
            The class index

        k_i: torch.tensor
            Ith part-count
        
        Returns
        -------
        h_i, w_i: torch.tensor, torch.tensor
            Ith height and width
        '''

        cd_h_class = self.height_matrix[class_idx]
        cd_w_class = self.width_matrix[class_idx]

        cd_h_dist = cd_h_class[k_i]
        cd_w_dist = cd_w_class[k_i]

        dist_h = Categorical(cd_h_dist)
        dist_w = Categorical(cd_w_dist)

        return dist_h.sample(), dist_w.sample()

    def forward(self, class_idx=None):
        '''
        Generate image type

        Parameters
        ----------
        class_idx: int
            The class index

        Returns
        -------
        phw_list: [(torch.tensor, torch.tensor, torch.tensor)]
            A list containing the feature index, the row, and the column of each part
            Needed for convolving over the final image
        '''

        if class_idx is not None:
            assert class_idx <= self.num_classes - 1, \
                "Class index is out of bounds"
        
        else:
            class_priors = torch.ones((self.num_classes)) / self.num_classes
            dist = Categorical(class_priors)
            class_idx = dist.sample()

        k = self.sample_k(class_idx)
        phw_list = []

        for i in range(k):
            part_i = self.sample_part(class_idx, i)
            h_i, w_i = self.sample_coordinate(class_idx, i)
            phw_list.append((part_i, h_i, w_i))

        if len(phw_list) == 0:
            self.forward(class_idx)

        return np.array(phw_list)