from typing import Dict

import numpy as np
import torch
from allennlp.data.fields import Field
from overrides import overrides
from PIL import Image
from torchvision.transforms import Compose


class EntityField(Field[np.array]):
    """
    An ``ImageField`` stores an image as a ``np.ndarray`` which must have exactly three
    dimensions.

    Adapted from https://github.com/sethah/allencv/blob/master/allencv/data/fields/image_field.py

    Parameters
    ----------
    image: ``np.ndarray``
    """

    def __init__(self,
                 entity_features: List,
                 padding_value: int = 0) -> None:

        self.padding_value = padding_value
        self.entity_features = torch.tensor(entity_features)

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        return self.entity_features

    @overrides
    def empty_field(self):  # pylint: disable=no-self-use
        return EntityField(np.empty(self.entity_features.shape),
                          padding_value=self.padding_value)

    def __str__(self) -> str:
        return f"ImageField with shape: {self.entity_features.shape}."
