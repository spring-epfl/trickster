import numpy as np
import keras
from keras import backend as K


class SaliencyOracle:
    '''
    Compute the saliency map of a Keras model for different examples

    Provides per-feature saliency values showing features that maximize
    classifier score for the target class.
    '''

    def __init__(self, model, target_class):
        '''
        Build saliency "oracle"

        :param model: Keras classifier
        :param target_class: target class, saliency for which is computed

        .. note::
           Only feed-forward networks with single input are supported.
           The first layer can be an embedding layer.
        '''
        self._model = model
        self._target_class = target_class

        self._input_tensor = input_tensor = model.input

        # If the first layer is an embedding, the forward derivative is taken
        # wrt to the embedding tensor.
        embed_layer = None
        if isinstance(model.layers[0], keras.layers.Embedding):
            embed_layer = model.layers[0]
            var_tensor = embed_layer(input_tensor)
            layers = model.layers[1:]

        # Otherwise, the forward derivative is taken wrt the model input.
        else:
            var_tensor = input_tensor
            layers = model.layers

        # Compute the output tensor, with dependence on the differentiation
        # variable (model input or embedding).
        # NOTE: This is the part that assumes the network is feed-forward.
        output_tensor = var_tensor
        for layer in layers:
            output_tensor = layer(output_tensor)

        adv_direction_grads, = K.gradients(output_tensor[:, target_class], [var_tensor])
        grads_sum, = K.gradients(output_tensor, [var_tensor])

        # Sum the gradients along the embedding dimension
        if embed_layer:
            adv_direction_grads = K.sum(adv_direction_grads, axis=-1)
            grads_sum = K.sum(grads_sum, axis=-1)

        # Compute the JSMA saliency map (Papernot et al., "The limitations
        # of deep learning in adversarial settings", 2016).
        other_grads_sum = grads_sum - adv_direction_grads
        mask = K.cast(adv_direction_grads > 0, 'float32') \
             * K.cast(other_grads_sum < 0, 'float32')
        self._saliency_tensor = mask \
                * adv_direction_grads * K.abs(other_grads_sum)

    def eval(self, examples):
        '''
        Evaluate saliency map for given examples

        :param examples: list of examples
        :return: array of shape (n, m), where n is the number of given
                 examples, and m is the input dimension.
        '''
        sess = K.get_session()
        saliency = sess.run(self._saliency_tensor, feed_dict={
            self._input_tensor: examples,
            keras.backend.learning_phase(): 0
        })
        return np.squeeze(saliency)
