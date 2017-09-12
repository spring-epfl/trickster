import numpy as np
import keras
from keras import backend as K


class SaliencyOracle:
    def __init__(self, model, target_class):
        '''
        :param model: Keras classifier

        Note:
            Only feed-forward networks with single input are supported.
            The first layer can be an embedding layer.
        '''
        input_tensor = model.input

        # If the first layer is embedding, compute forward derivative
        # w.r.t to the embedding.
        embed_layer = None
        if isinstance(model.layers[0], keras.layers.Embedding):
            embed_layer = model.layers[0]
            var_tensor = embed_layer(input_tensor)
            layers = model.layers[1:]

        # Otherwise, the forward derivative is computer w.r.t to the input
        else:
            var_tensor = input_tensor
            layers = model.layers

        # Compute the output tensor, that depends on the differentiation variable
        # tensor
        # NOTE: This is the part that assumes the network is feed-forward.
        output_tensor = var_tensor
        for layer in layers:
            output_tensor = layer(output_tensor)

        # Compute saliency map
        adv_direction_grads, = K.gradients(output_tensor[:, target_class], [var_tensor])
        grads_sum, = K.gradients(output_tensor, [var_tensor])

        # Sum the gradients along the embedding dimension
        if embed_layer:
            adv_direction_grads = K.sum(adv_direction_grads, axis=-1)
            grads_sum = K.sum(grads_sum, axis=-1)

        other_grads_sum = grads_sum - adv_direction_grads
        relevant_mask = K.cast(adv_direction_grads > 0, 'float32') \
                      * K.cast(other_grads_sum < 0, 'float32')
        saliency = relevant_mask * adv_direction_grads * K.abs(other_grads_sum)

        self.input_tensor = input_tensor
        self.saliency_tensor = saliency

    def eval(self, examples):
        sess = K.get_session()
        saliency = np.squeeze(sess.run(self.saliency_tensor, feed_dict={
            self.input_tensor: examples,
            keras.backend.learning_phase(): 0
        }))
        return saliency
