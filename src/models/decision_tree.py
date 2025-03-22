from typing import List, Dict, Set
from collections import Counter
import math
from .node import Node

class DecisionTree:
    def __init__(self):
        self.root = None
        self.attributes = {}
        self.class_attribute = None
        self.class_values = set()
        
    def fit(self, data: List[List[str]], attributes: Dict[str, List[str]], 
            class_attribute: str, class_values: Set[str]) -> None:
        """Build the decision tree using the ID3 algorithm."""
        self.attributes = attributes
        self.class_attribute = class_attribute
        self.class_values = class_values
        
        # Convert data to list of dictionaries for easier processing
        data_dict = self._convert_to_dict(data)
        
        # Build the tree
        self.root = self._build_tree(data_dict, list(attributes.keys()))
        
    def _convert_to_dict(self, data: List[List[str]]) -> List[Dict[str, str]]:
        """Convert list of lists to list of dictionaries."""
        result = []
        for row in data:
            # Create a dictionary with all attributes including the class attribute
            row_dict = dict(zip(list(self.attributes.keys()) + [self.class_attribute], row))
            result.append(row_dict)
        return result
    
    def _build_tree(self, data: List[Dict[str, str]], 
                   available_attributes: List[str]) -> Node:
        """Recursively build the decision tree using ID3 algorithm."""
        if not data:
            return None
            
        # If all examples have the same class, create a leaf node
        classes = [row[self.class_attribute] for row in data]
        if len(set(classes)) == 1:
            node = Node()
            node.set_as_leaf(classes[0])
            return node
            
        # If no attributes left, create a leaf node with majority class
        if not available_attributes:
            node = Node()
            majority_class = Counter(classes).most_common(1)[0][0]
            node.set_as_leaf(majority_class)
            return node
            
        # Find the best attribute to split on
        best_attribute = self._select_best_attribute(data, available_attributes)
        
        # Create a node for the best attribute
        node = Node(attribute=best_attribute)
        
        # For each value of the best attribute
        for value in self.attributes[best_attribute]:
            # Get examples where attribute = value
            subset = [row for row in data if row[best_attribute] == value]
            
            if subset:
                # Recursively build subtree
                remaining_attributes = [attr for attr in available_attributes 
                                     if attr != best_attribute]
                child = self._build_tree(subset, remaining_attributes)
                
                if child:
                    node.add_child(value, child)
            else:
                # Create leaf node with majority class
                child = Node()
                majority_class = Counter(classes).most_common(1)[0][0]
                child.set_as_leaf(majority_class)
                node.add_child(value, child)
                
        return node
    
    def _select_best_attribute(self, data: List[Dict[str, str]], 
                             attributes: List[str]) -> str:
        """Select the best attribute to split on using information gain."""
        best_gain = -float('inf')
        best_attribute = None
        
        for attribute in attributes:
            gain = self._information_gain(data, attribute)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute
                
        return best_attribute
    
    def _information_gain(self, data: List[Dict[str, str]], attribute: str) -> float:
        """Calculate information gain for an attribute."""
        # Calculate entropy before split
        classes = [row[self.class_attribute] for row in data]
        entropy_before = self._entropy(classes)
        
        # Calculate weighted entropy after split
        entropy_after = 0
        for value in self.attributes[attribute]:
            subset = [row for row in data if row[attribute] == value]
            if subset:
                subset_classes = [row[self.class_attribute] for row in subset]
                entropy_after += (len(subset) / len(data)) * self._entropy(subset_classes)
                
        return entropy_before - entropy_after
    
    def _entropy(self, classes: List[str]) -> float:
        """Calculate entropy for a list of classes."""
        if not classes:
            return 0
            
        counts = Counter(classes)
        total = len(classes)
        entropy = 0
        
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log2(probability)
            
        return entropy
    
    def predict(self, instance: Dict[str, str]) -> str:
        """Predict the class for a given instance."""
        if not self.root:
            raise ValueError("Tree not trained yet")
            
        current = self.root
        while not current.is_leaf_node():
            attribute = current.attribute
            value = instance[attribute]
            current = current.get_child(value)
            
        return current.get_prediction()
    
    def evaluate(self, test_data: List[Dict[str, str]]) -> float:
        """Evaluate the tree on test data and return accuracy."""
        correct = 0
        total = len(test_data)
        
        for instance in test_data:
            predicted = self.predict(instance)
            actual = instance[self.class_attribute]
            if predicted == actual:
                correct += 1
                
        return correct / total if total > 0 else 0 