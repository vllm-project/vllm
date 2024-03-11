__author__ = 'Alex Rogozhnikov'

from ..layers.tensorflow import Rearrange, Reduce, EinMix

keras_custom_objects = {
    Rearrange.__name__: Rearrange,
    Reduce.__name__: Reduce,
    EinMix.__name__: EinMix,
}
