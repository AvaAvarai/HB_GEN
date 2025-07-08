import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import os

def download_mnist():
    """
    Download MNIST dataset from sklearn.
    Returns:
        X: images as numpy array
        y: labels as numpy array
    """
    print("Downloading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target
    print(f"Downloaded {len(X)} images with shape {X.shape}")
    return X, y

def reshape_images(X):
    """
    Reshape flattened images back to 28x28.
    
    Args:
        X: flattened images (n_samples, 784)
    
    Returns:
        reshaped images (n_samples, 28, 28)
    """
    return X.reshape(-1, 28, 28)

def remove_edges(images, pixels=3):
    """
    Remove specified number of pixels from each edge of images.
    
    Args:
        images: numpy array of shape (n_samples, height, width)
        pixels: number of pixels to remove from each edge
    
    Returns:
        cropped images
    """
    return images[:, pixels:-pixels, pixels:-pixels]

def apply_2x2_pooling(images):
    """
    Apply 2x2 average pooling to images.
    
    Args:
        images: numpy array of shape (n_samples, height, width)
    
    Returns:
        pooled images
    """
    n_samples, height, width = images.shape
    pooled_height = height // 2
    pooled_width = width // 2
    
    # Reshape to apply pooling
    pooled_images = np.zeros((n_samples, pooled_height, pooled_width))
    
    for i in range(pooled_height):
        for j in range(pooled_width):
            # Extract 2x2 block and compute average
            block = images[:, 2*i:2*i+2, 2*j:2*j+2]
            pooled_images[:, i, j] = np.mean(block, axis=(1, 2))
    
    return pooled_images

def flatten_images(images):
    """
    Flatten images from (n_samples, height, width) to (n_samples, height*width).
    
    Args:
        images: numpy array of shape (n_samples, height, width)
    
    Returns:
        flattened images
    """
    return images.reshape(images.shape[0], -1)

def process_mnist_dataset():
    """
    Main function to process MNIST dataset according to specifications.
    Separates training (60,000) and test (10,000) sets.
    """
    print("Starting MNIST dataset processing...")
    
    # Step 1: Download MNIST dataset
    X, y = download_mnist()
    
    # Step 2: Split into training and test sets
    print("Splitting into training and test sets...")
    # MNIST is already split: first 60,000 are training, last 10,000 are test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 3: Process training set
    print("\nProcessing training set...")
    X_train_reshaped = reshape_images(X_train)
    X_train_cropped = remove_edges(X_train_reshaped, pixels=3)
    X_train_pooled = apply_2x2_pooling(X_train_cropped)
    X_train_flat = flatten_images(X_train_pooled)
    
    # Step 4: Process test set
    print("Processing test set...")
    X_test_reshaped = reshape_images(X_test)
    X_test_cropped = remove_edges(X_test_reshaped, pixels=3)
    X_test_pooled = apply_2x2_pooling(X_test_cropped)
    X_test_flat = flatten_images(X_test_pooled)
    
    print(f"Training shape after processing: {X_train_flat.shape}")
    print(f"Test shape after processing: {X_test_flat.shape}")
    
    # Step 5: Create DataFrames with features first, class last
    print("Creating DataFrames...")
    feature_names = [f'pixel_{i}' for i in range(X_train_flat.shape[1])]
    
    # Training DataFrame
    df_train = pd.DataFrame(X_train_flat, columns=feature_names)
    df_train['class'] = y_train
    
    # Test DataFrame
    df_test = pd.DataFrame(X_test_flat, columns=feature_names)
    df_test['class'] = y_test
    
    # Step 6: Save to CSV files
    print("Saving datasets...")
    
    # Create datasets directory if it doesn't exist
    os.makedirs('datasets', exist_ok=True)
    
    # Save training set
    train_file = 'datasets/mnist_dr_train.csv'
    df_train.to_csv(train_file, index=False)
    print(f"✓ Training set saved to {train_file}")
    
    # Save test set
    test_file = 'datasets/mnist_dr_test.csv'
    df_test.to_csv(test_file, index=False)
    print(f"✓ Test set saved to {test_file}")
    
    print(f"\n✓ Successfully processed and saved MNIST datasets!")
    print(f"  - Original dimensions: 784 features (28x28)")
    print(f"  - After cropping (22x22): 484 features")
    print(f"  - After 2x2 pooling (11x11): 121 features")
    print(f"  - Training samples: {len(df_train)}")
    print(f"  - Test samples: {len(df_test)}")
    print(f"  - Training file: {train_file}")
    print(f"  - Test file: {test_file}")
    
    # Display first few rows of training set
    print("\nFirst few rows of the training dataset:")
    print(df_train.head())
    
    return df_train, df_test

if __name__ == "__main__":
    # Process the dataset
    train_df, test_df = process_mnist_dataset() 