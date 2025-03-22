try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
from typing import Dict, Set, List, Tuple
from models.node import Node

def visualize_tree_text(root: Node) -> str:
    """Generate a text-based visualization of the tree."""
    lines = []
    
    def _add_node(node, prefix="", is_last=True, is_value=False):
        # Characters for drawing tree
        BOX_VERTICAL = "│"
        BOX_HORIZONTAL = "─"
        BOX_VERTICAL_RIGHT = "├"
        BOX_UP_RIGHT = "└"
        
        # Add current node line
        if node.is_leaf_node():
            # It's a leaf node with class prediction
            conn = BOX_UP_RIGHT if is_last else BOX_UP_RIGHT
            lines.append(f"{prefix}{conn}{BOX_HORIZONTAL}{BOX_HORIZONTAL} Class: {node.get_prediction()}")
            return
            
        # It's a decision node with an attribute
        if not is_value:  # Skip if this is a value line
            conn = BOX_UP_RIGHT if is_last else BOX_VERTICAL_RIGHT
            lines.append(f"{prefix}{conn}{BOX_HORIZONTAL}{BOX_HORIZONTAL} {node.attribute}")
        
        # Process children
        child_items = list(node.children.items())
        
        # Indentation for child nodes
        child_prefix = prefix + ("    ")
        
        # Add each child with its branch
        for i, (value, child) in enumerate(child_items):
            is_last_child = i == len(child_items) - 1
            
            # Add the value branch line
            conn = BOX_UP_RIGHT if is_last_child else BOX_VERTICAL_RIGHT
            lines.append(f"{child_prefix}{conn}{BOX_HORIZONTAL}{BOX_HORIZONTAL} {value}")
            
            # Next level prefix (add vertical line if not last)
            next_prefix = child_prefix + ("    " if is_last_child else f"{BOX_VERTICAL}   ")
            
            
            # Process the child node
            _add_node(child, next_prefix, is_last_child, is_value=False)
    
    # Start the recursion from the root
    _add_node(root)
    return "\n".join(lines)

def save_text_visualization(tree, output_file: str) -> None:
    """Save the text-based visualization to a file."""
    if not tree.root:
        return
        
    header = "Decision Tree Visualization\n" + "=" * 50 + "\n\n"
    text_tree = header + visualize_tree_text(tree.root)
    
    with open(f"{output_file}.txt", 'w', encoding='utf-8') as f:
        f.write(text_tree)
    print(f"Text tree visualization saved to {output_file}.txt")

def visualize_tree(tree, attributes: Dict[str, list], class_attribute: str, 
                  class_values: Set[str], output_file: str = None) -> None:
    """Visualize the decision tree using Graphviz and/or text-based visualization."""
    if not tree.root:
        return
        
    # Always generate text visualization
    if output_file:
        save_text_visualization(tree, output_file)
    
    # Generate Graphviz visualization if available
    if GRAPHVIZ_AVAILABLE:
        dot = graphviz.Digraph(comment='Decision Tree')
        dot.attr(rankdir='TB')  # Top to Bottom direction
        
        def add_node(node: Node, parent_id: str = None, edge_label: str = None) -> str:
            """Recursively add nodes to the graph."""
            if node.is_leaf_node():
                # Create leaf node
                node_id = f"leaf_{id(node)}"
                label = f"Class: {node.get_prediction()}"
                dot.node(node_id, label, shape='box', style='filled', fillcolor='lightgreen')
            else:
                # Create decision node
                node_id = f"node_{id(node)}"
                label = f"{node.attribute}"
                dot.node(node_id, label)
                
            # Add edge from parent if exists
            if parent_id is not None and edge_label is not None:
                dot.edge(parent_id, node_id, edge_label)
                
            # Recursively add children
            if not node.is_leaf_node():
                for value, child in node.children.items():
                    add_node(child, node_id, value)
                    
            return node_id
        
        # Start visualization from root
        add_node(tree.root)
        
        # Save the visualization
        if output_file:
            dot.render(output_file, format='png', cleanup=True)
            print(f"Graph visualization saved to {output_file}.png")
        else:
            # Display the tree
            dot.view()
    else:
        print("Graphviz is not available. Please install it using:")
        print("pip install graphviz")
        print("And make sure the Graphviz system package is installed:")
        print("- On Windows: Download and install from https://graphviz.org/download/")
        print("- On Linux: sudo apt-get install graphviz")
        print("- On macOS: brew install graphviz")
        print("\nA text-based visualization has been saved instead.") 