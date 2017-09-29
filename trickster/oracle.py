import numpy as np
import keras
from keras import backend as K


class SaliencyOracle:
    '''
    Compute the saliency map of a Keras model for different examples

    Provides per-feature saliency values showing features that maximize
    classifier score for the target class.
    '''

    def __init__(self, model, target_class, kind='positive'):
        '''
        Build saliency "oracle"

        :param model: Keras classifier.
        :param target_class: target class, saliency for which is computed.
        :param positive: kind of saliency map. Only 'positive' is
                supported atm.

        .. note::
           Only feed-forward networks with single input are supported.
           The first layer can be an embedding layer.
        '''
        self.model = model
        self.target_class = target_class
        self.kind = kind

    def get_input_tensor(self):
        return self.model.input

    def get_saliency_tensor(self, featurewise=True):
        # If the first layer is an embedding, the forward derivative is taken
        # wrt to the embedding tensor.
        embed_layer = None
        input_tensor = self.get_input_tensor()
        if isinstance(self.model.layers[0], keras.layers.Embedding):
            embed_layer = self.model.layers[0]
            var_tensor = embed_layer(input_tensor)
            layers = self.model.layers[1:]

        # Otherwise, the forward derivative is taken wrt the self.model input.
        else:
            var_tensor = input_tensor
            layers = self.model.layers

        # Compute the output tensor, with dependence on the differentiation
        # variable (self.model input or embedding).
        # NOTE: This is the part that assumes the network is feed-forward.
        output_tensor = var_tensor
        for layer in layers:
            output_tensor = layer(output_tensor)

        adv_direction_grads, = K.gradients(output_tensor[:, self.target_class],
                                           [var_tensor])
        grads_sum, = K.gradients(output_tensor, [var_tensor])

        # Sum the gradients along the embedding dimension
        if embed_layer:
            adv_direction_grads = K.sum(adv_direction_grads, axis=-1)
            grads_sum = K.sum(grads_sum, axis=-1)

        # Compute the JSMA saliency map (Papernot et al., "The limitations
        # of deep learning in adversarial settings", 2016).
        if self.kind == 'positive':
            other_grads_sum = grads_sum - adv_direction_grads
            mask = K.cast(adv_direction_grads > 0, 'float32') \
                 * K.cast(other_grads_sum < 0, 'float32')

            saliency_tensor = mask * adv_direction_grads * K.abs(other_grads_sum)

        else:
            raise ValueError('Unsupported kind: %s.' % self.kind)

        if not featurewise:
            saliency_tensor = K.sum(saliency_tensor, axis=1)
        return saliency_tensor

    def eval(self, examples, featurewise=True):
        '''
        Evaluate saliency map for given examples

        :param examples: list of examples.
        :param featurewise: whether to report saliency per feature (i.e. input
                dimension), or sum it up over all dimensions.
        :return: array of shape (n, m), where n is the number of given
                examples, and m is the input dimension, or of shape
                (n) if ``featurewise`` is set to ``False``.

        '''
        if not hasattr(self, '_saliency_tensor'):
            self._saliency_tensor = self.get_saliency_tensor(featurewise)

        sess = K.get_session()
        saliency = sess.run(self._saliency_tensor, feed_dict={
            self.get_input_tensor(): examples,
            keras.backend.learning_phase(): 0
        })
        return saliency
