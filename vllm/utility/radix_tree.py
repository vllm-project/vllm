class InvalidKeyError(Exception):
    """Custom exception for invalid key operations."""
    pass

class RadixTreeNode:
    def __init__(self, value:str="", data:memory_path=None):
        self.value = value
        self.data = data
        self.children: dict[str, RadixTreeNode] = {}

class RadixTree:
    def __init__(self, trunk_size:int=1):
        self.root = RadixTreeNode()
        self.trunk_size = trunk_size  # Dynamically adjustable trunk size

    def find(self, key: str) -> list[RadixTreeNode]:
        if not key:
            raise InvalidKeyError("Empty key is not allowed.")

        current_node: RadixTreeNode = self.root
        exist_nodes_to_return: list[RadixTreeNode] = []
        i = 0

        while i < len(key):
            chunk = key[i:i + self.trunk_size]
            i += self.trunk_size

            if chunk in current_node.children:
                current_node = current_node.children[chunk]
                if current_node.data is not None:  # Consider nodes with data
                    exist_nodes_to_return.append(current_node)
            else:
                break  # Chunk not found, stop traversal

        return exist_nodes_to_return
    
    def insert(self, key: str, data: list) -> None:
        if not key:
            raise InvalidKeyError("Key cannot be empty.")
        if not data:
            raise ValueError("Data list cannot be empty.")

        current_node = self.root
        token_index = 0  # Index to track the current token's data in the list
        
        while token_index < len(data):
            # Calculate the current position in the key based on the token index and trunk size
            i = token_index * self.trunk_size
            # Determine the chunk of the key for the current token(s)
            chunk = key[i:i + self.trunk_size]
            data_chunk = data[i:i + self.trunk_size]  # Get the data for the current token(s)

            if not chunk:  # If the chunk is empty, break the loop
                break
            
            if chunk not in current_node.children:
                current_node.children[chunk] = RadixTreeNode(chunk, data_chunk)
            else:
                # If the chunk already exists, we update the data if it's not set
                if current_node.children[chunk].data is None:
                    current_node.children[chunk].data = data_chunk
            
            current_node = current_node.children[chunk]
            token_index += 1