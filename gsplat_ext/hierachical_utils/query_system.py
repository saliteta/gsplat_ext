import torch
from typing import List, Union, Literal, Tuple

class QuerySystem:
    def __init__(self, number_of_layers: int):
        self.query_model = None
        self.number_of_layers = number_of_layers

    def query(self, query_text: Union[str, None] = None)-> Union[List[torch.Tensor], None]:
        """
            Query the system with a single text.
            The example of text is: "00" or "001"
            00 means query from top to bottom, layer 2, node 0 and all its decendents
            001 means query from top to bottom, layer 3, node 1 and all its decendents
            We need to return a list of torch tensors determine the index range of the nodes in each layer
            The length of the list is the number of layers.
            Higher layer has no index, therefore return None
            Lower layer has index, therefore return the index range of the nodes in the layer

            Args:
                query_text: the text to query the system
            Returns:
                A list of torch tensors determine the index range of the nodes in each layer
        """
        if query_text is not None:
            assert isinstance(query_text, str), f"Query text must be a string or None, but got {type(query_text)}"
            assert all(c in "01" for c in query_text), f"Query text must only contain '0' and '1', but got {query_text}"
            assert len(query_text) < self.number_of_layers, f"Query text length must be less than the number of layers, but got {len(query_text)}"
        else:
            return None
        highest_layer_number = len(query_text)
        layer_index = int(query_text, 2)
        index_range = []
        for layer in range(highest_layer_number):
            index_range.append(None)
        for layer in range(highest_layer_number, self.number_of_layers):
            index_range.append(torch.tensor([layer_index << (layer - highest_layer_number), ((layer_index+1) << (layer - highest_layer_number)) - 1]))
        return index_range

