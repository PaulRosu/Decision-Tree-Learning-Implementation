import random
from typing import List, Tuple

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.attributes = {}
        self.data = []
        self.class_attribute = None
        self.class_values = set()
        
    def load_data(self) -> None:
        """Load and parse the ARFF file."""
        with open(self.file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
            
        # Parse attributes
        data_start = -1
        attribute_lines = []
        
        # First pass: collect all attribute lines and find data start
        for i, line in enumerate(lines):
            # Skip empty lines and comments
            if not line or line.startswith('%'):
                continue
                
            # Convert to lowercase for case-insensitive comparison
            line_lower = line.lower()
            
            if line_lower.startswith('@attribute'):
                attribute_lines.append((i, line))
            elif line_lower.startswith('@data'):
                data_start = i + 1
                break
        
        if data_start == -1:
            raise ValueError("No @data section found in ARFF file")
            
        # Second pass: process attributes
        for i, line in attribute_lines:
            # Skip empty lines and comments
            if not line or line.startswith('%'):
                continue
                
            parts = line.split()
            if len(parts) >= 3:
                attr_name = parts[1]
                # Check if this is the class attribute (last attribute before @data)
                if i == attribute_lines[-1][0]:
                    self.class_attribute = attr_name
                    # Parse class values from curly braces
                    values_str = line[line.find('{')+1:line.find('}')]
                    self.class_values = set(v.strip() for v in values_str.split(','))
                else:
                    # Parse attribute values from curly braces
                    values_str = line[line.find('{')+1:line.find('}')]
                    self.attributes[attr_name] = [v.strip() for v in values_str.split(',')]
        
        if not self.class_attribute:
            raise ValueError("No class attribute found in ARFF file")
        
        # Parse data
        for line in lines[data_start:]:
            # Skip empty lines and comments
            if not line or line.startswith('%'):
                continue
                
            values = [v.strip() for v in line.split(',')]
            if len(values) == len(self.attributes) + 1:  # +1 for class attribute
                self.data.append(values)
                
    def split_data(self, train_ratio: float = 0.7) -> Tuple[List[List[str]], List[List[str]]]:
        """Split the dataset into training and testing sets."""
        if not self.data:
            self.load_data()
            
        # Shuffle the data
        random.shuffle(self.data)
        
        # Calculate split point
        split_idx = int(len(self.data) * train_ratio)
        
        # Split the data
        train_data = self.data[:split_idx]
        test_data = self.data[split_idx:]
        
        return train_data, test_data
    
    def save_split(self, train_data: List[List[str]], test_data: List[List[str]], 
                  train_file: str, test_file: str) -> None:
        """Save the split datasets to files."""
        # Get the relation name from the original file
        relation_name = "dataset"  # default name
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip().lower()
                if line.startswith('@relation'):
                    relation_name = line.split()[1]
                    break
        
        # Save training data
        with open(train_file, 'w') as f:
            # Write header
            f.write(f'@relation {relation_name}\n\n')
            for attr, values in self.attributes.items():
                f.write(f'@attribute {attr} {{{",".join(values)}}}\n')
            f.write(f'@attribute {self.class_attribute} {{{",".join(self.class_values)}}}\n\n')
            f.write('@data\n')
            # Write data
            for row in train_data:
                f.write(','.join(row) + '\n')
                
        # Save testing data
        with open(test_file, 'w') as f:
            # Write header
            f.write(f'@relation {relation_name}\n\n')
            for attr, values in self.attributes.items():
                f.write(f'@attribute {attr} {{{",".join(values)}}}\n')
            f.write(f'@attribute {self.class_attribute} {{{",".join(self.class_values)}}}\n\n')
            f.write('@data\n')
            # Write data
            for row in test_data:
                f.write(','.join(row) + '\n') 