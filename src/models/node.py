from typing import Optional, Dict

class Node:
    def __init__(self, attribute: Optional[str] = None, value: Optional[str] = None):
        self.attribute = attribute  # The attribute to split on
        self.value = value        # The value of the attribute for this branch
        self.children: Dict[str, 'Node'] = {}  # Child nodes for each attribute value
        self.is_leaf = False      # Whether this is a leaf node
        self.prediction = None    # The class prediction for leaf nodes
        
    def add_child(self, value: str, child: 'Node') -> None:
        """Add a child node for a specific attribute value."""
        self.children[value] = child
        
    def get_child(self, value: str) -> Optional['Node']:
        """Get the child node for a specific attribute value."""
        return self.children.get(value)
    
    def set_as_leaf(self, prediction: str) -> None:
        """Set this node as a leaf node with a prediction."""
        self.is_leaf = True
        self.prediction = prediction
        
    def is_leaf_node(self) -> bool:
        """Check if this is a leaf node."""
        return self.is_leaf
    
    def get_prediction(self) -> Optional[str]:
        """Get the prediction of this node (if it's a leaf node)."""
        return self.prediction if self.is_leaf else None 