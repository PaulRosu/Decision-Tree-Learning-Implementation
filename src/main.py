import os
import argparse
import pickle
from data.data_loader import DataLoader
from models.decision_tree import DecisionTree
from utils.visualization import visualize_tree

def find_arff_file(keyword: str) -> str:
    """Find the first ARFF file containing the keyword in its name."""
    raw_dir = 'data/raw'
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Directory {raw_dir} not found")
        
    for file in os.listdir(raw_dir):
        if file.endswith('.arff') and keyword.lower() in file.lower():
            return os.path.join(raw_dir, file)
    
    raise FileNotFoundError(f"No ARFF file found containing keyword '{keyword}'")

def save_model(tree: DecisionTree, dataset_name: str) -> None:
    """Save the trained model to a file."""
    model_path = f'data/models/{dataset_name}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(tree, f)
    print(f"Model saved to {model_path}")

def load_model(dataset_name: str) -> DecisionTree:
    """Load a trained model from a file."""
    model_path = f'data/models/{dataset_name}_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model found at {model_path}")
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def process_dataset(file_path: str, dataset_name: str, mode: str = 'all', visualize: bool = False) -> None:
    """Process a single dataset based on the specified mode."""
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Initialize data loader
    data_loader = DataLoader(file_path)
    
    # Load and split the data
    train_data, test_data = data_loader.split_data(train_ratio=0.7)
    
    # Save the split datasets if needed
    if mode in ['all', 'split']:
        data_loader.save_split(
            train_data, 
            test_data,
            f'data/splits/{dataset_name}_train.arff',
            f'data/splits/{dataset_name}_test.arff'
        )
        print(f"Split datasets saved to data/splits/{dataset_name}_*.arff")
    
    # Train or load the model if needed
    if mode in ['all', 'train', 'test']:
        try:
            if mode == 'test':
                # Try to load existing model for testing
                tree = load_model(dataset_name)
                print(f"Loaded existing model for {dataset_name}")
            else:
                # Train new model
                tree = DecisionTree()
                tree.fit(
                    train_data,
                    data_loader.attributes,
                    data_loader.class_attribute,
                    data_loader.class_values
                )
                print(f"Model trained successfully")
                # Save the model if we trained it
                if mode in ['all', 'train']:
                    save_model(tree, dataset_name)
        except FileNotFoundError:
            print(f"No saved model found for {dataset_name}. Training new model...")
            tree = DecisionTree()
            tree.fit(
                train_data,
                data_loader.attributes,
                data_loader.class_attribute,
                data_loader.class_values
            )
            print(f"Model trained successfully")
            save_model(tree, dataset_name)
    
    # Evaluate the model if needed
    if mode in ['all', 'test']:
        # Convert test data to dictionary format for evaluation
        test_data_dict = []
        for row in test_data:
            # Convert dict_keys to list before concatenation
            attribute_names = list(data_loader.attributes.keys()) + [data_loader.class_attribute]
            test_data_dict.append(dict(zip(attribute_names, row)))
        
        # Evaluate the tree
        accuracy = tree.evaluate(test_data_dict)
        print(f"\nDecision Tree Results for {dataset_name}:")
        print(f"Training set size: {len(train_data)}")
        print(f"Testing set size: {len(test_data)}")
        print(f"Accuracy on test set: {accuracy:.2%}")
    
    # Visualize the tree if requested (moved outside of test block)
    if visualize:
        output_file = f'data/models/{dataset_name}_tree'
        visualize_tree(
            tree,
            data_loader.attributes,
            data_loader.class_attribute,
            data_loader.class_values,
            output_file
        )

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Decision Tree Learning Implementation')
    parser.add_argument('--mode', choices=['all', 'split', 'train', 'test'], 
                      default='all', help='Operation mode: all (default), split, train, or test')
    parser.add_argument('--dataset', type=str,
                      help='Keyword to search in ARFF filenames (e.g., "weather" or "soybean")')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize the decision tree')
    args = parser.parse_args()
    
    # Create necessary directories if they don't exist
    os.makedirs('data/splits', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    
    try:
        if args.dataset:
            # Find and process the specified dataset
            file_path = find_arff_file(args.dataset)
            process_dataset(file_path, args.dataset, args.mode, args.visualize)
        else:
            # Process all ARFF files in the raw directory
            raw_dir = 'data/raw'
            if not os.path.exists(raw_dir):
                raise FileNotFoundError(f"Directory {raw_dir} not found")
                
            arff_files = [f for f in os.listdir(raw_dir) if f.endswith('.arff')]
            if not arff_files:
                raise FileNotFoundError("No ARFF files found in data/raw directory")
                
            for file in arff_files:
                dataset_name = os.path.splitext(file)[0]  # Remove .arff extension
                file_path = os.path.join(raw_dir, file)
                process_dataset(file_path, dataset_name, args.mode, args.visualize)
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 