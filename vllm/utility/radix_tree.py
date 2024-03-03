    """
    radix_tree.py
    
    
    """

class RadixTreeNode:
    def __init__(self, value:str="", data:memory_path=None):
        self.value = value
        self.data = data
        self.children = {}

class RadixTree:
    def __init__(self, trunk_size:int=1):
        self.root = RadixTreeNode()
        self.trunk_size = trunk_size  # Dynamically adjustable trunk size

    def find(self, key:str) -> RadixTreeNode:
        current_node = self.root
        current_key_inx = 0