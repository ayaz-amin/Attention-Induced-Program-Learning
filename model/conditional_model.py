from forward_model.model import AttentionInduction, ConditionalAttentionInduction

from inference_model.learning import train_image
from inference_model.inference import infer


class ConditionalTaskModel(object):
    '''
    This object executes models for conditional tasks (e.g. classification, conditional generation)

    Parameters
    ----------

    num_training_examples: int
        Number of training examples (samples from conditional generative model)
        to train the inference model
    num_classes: int
        Number of classes
    max_k: int
        Maximum number of sub-parts allowed to be sampled
    num_parts: int
        Number of sub-parts
    filter_size: (int, int)
        Filter size of the generative model
    image_shape: (int, int, int)
        Shape of the image
    '''

    def __init__(self, num_training_examples, num_classes, max_k, num_parts, filter_size, image_shape):
        '''
        Attributes
        ----------

        num_training_examples:
            The number of training examples (samples from the conditional generative model)
            to train the inference model
        num_classes: int
            The number of classes
        model: nn.Module
            The conditional generative model
        model_factors: [(numpy.ndarray, networkx.Graph)]
            Model factors for inference model
        '''

        self.num_training_examples = num_training_examples
        self.num_classes = num_classes

        self.model = ConditionalAttentionInduction(
            num_classes=num_classes,
            max_k=max_k,
            num_parts=num_parts,
            filter_size=filter_size,
            image_shape=image_shape
        )

        self.model_factors, self.labels = self.create_inference_model()

    def create_inference_model(self):
        '''
        Creates an inference model (Recursive Cortical Network) for bottom up proposals
        by training it on samples from the conditional generative model

        Returns
        -------

        model_factors: [(numpy.ndarray, networkx.Graph)]
            Model factors of the RCN in the form of a tuple containing
            the pooling centers and the graphs
        labels: [int]
            Corresponding labels, needed for running MCMC on the conditional generative model
        '''

        model_factors = []
        labels = []

        for i in range(self.num_classes):
            for _ in range(self.num_training_examples):
                sample = self.model(i)
                factors = train_image(sample)
                model_factors.append(factors)
                labels.append(i)

        return model_factors, labels

    def predict(self, image, pool_shape):
        '''
        Predict the class that the input image belongs to

        Parameters
        ----------

        image: numpy.ndarray
            The testing image
        pool_shape: (int, int)
            Pooling shape of the RCN

        Returns
        -------

        top_candidate: int
            The predicted index of the class
        '''

        # Forward pass
        top_candidate = infer(image, self.model_factors, pool_shape=pool_shape, num_candidates=1)
        return self.labels[top_candidate]

    def generate_new_exemplars(self, class_idx=None):
        '''
        Generate new exemplars of a given class

        Returns
        -------

        image: 2D torch.tensor
            The generated image
        '''
        
        return self.model(class_idx)