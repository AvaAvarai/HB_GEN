import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

def load_processed_mnist():
    """
    Load the processed MNIST dataset.
    
    Returns:
        features: numpy array of shape (n_samples, 121)
        labels: numpy array of shape (n_samples,)
    """
    print("Loading processed MNIST dataset...")
    df = pd.read_csv('datasets/mnist_dr.csv')
    
    # Separate features and labels
    feature_columns = [col for col in df.columns if col != 'class']
    features = df[feature_columns].values
    labels = df['class'].values
    
    print(f"Loaded {len(features)} samples with {features.shape[1]} features")
    return features, labels

def reshape_to_image(features):
    """
    Reshape flattened features back to 11x11 images.
    
    Args:
        features: numpy array of shape (n_samples, 121)
    
    Returns:
        images: numpy array of shape (n_samples, 11, 11)
    """
    return features.reshape(-1, 11, 11)

def create_atlas(images, labels, n_images=81, grid_size=9):
    """
    Create a 9x9 atlas of the first 81 processed MNIST images.
    
    Args:
        images: numpy array of shape (n_samples, 11, 11)
        labels: numpy array of shape (n_samples,)
        n_images: number of images to display (default: 81)
        grid_size: size of the grid (default: 9 for 9x9)
    """
    # Take the first n_images
    images_subset = images[:n_images]
    labels_subset = labels[:n_images]
    
    # Create figure and grid
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(grid_size, grid_size, figure=fig, wspace=0.1, hspace=0.1)
    
    print(f"Creating {grid_size}x{grid_size} atlas with first {n_images} images...")
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            
            if idx < n_images:
                # Get the image and label
                img = images_subset[idx]
                label = labels_subset[idx]
                
                # Create subplot
                ax = fig.add_subplot(gs[i, j])
                
                # Display image
                im = ax.imshow(img, cmap='gray', interpolation='nearest')
                ax.set_title(f'Digit {label}', fontsize=10, pad=5)
                
                # Remove axes
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                # Add border
                rect = patches.Rectangle((0, 0), 11, 11, linewidth=1, 
                                       edgecolor='black', facecolor='none')
                ax.add_patch(rect)
                
            else:
                # Empty subplot for remaining grid positions
                ax = fig.add_subplot(gs[i, j])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_facecolor('lightgray')
    
    # Add overall title
    fig.suptitle('MNIST Atlas: First 81 Processed Images (11×11 after cropping and pooling)', 
                 fontsize=16, y=0.98)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Pixel Intensity', rotation=270, labelpad=15)
    
    plt.tight_layout()
    return fig

def create_comparison_atlas(original_images, processed_images, labels, n_images=81, grid_size=9):
    """
    Create a comparison atlas showing original vs processed images.
    
    Args:
        original_images: numpy array of original 28x28 images
        processed_images: numpy array of processed 11x11 images
        labels: numpy array of labels
        n_images: number of images to display
        grid_size: size of the grid
    """
    # Take the first n_images
    orig_subset = original_images[:n_images]
    proc_subset = processed_images[:n_images]
    labels_subset = labels[:n_images]
    
    # Create figure with two rows
    fig, axes = plt.subplots(2, grid_size, figsize=(20, 8))
    fig.suptitle('MNIST Comparison: Original (28×28) vs Processed (11×11)', fontsize=16)
    
    print(f"Creating comparison atlas with first {n_images} images...")
    
    for i in range(grid_size):
        if i < n_images:
            # Original image (top row)
            ax_orig = axes[0, i]
            im_orig = ax_orig.imshow(orig_subset[i], cmap='gray', interpolation='nearest')
            ax_orig.set_title(f'Original {labels_subset[i]}', fontsize=10)
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
            
            # Processed image (bottom row)
            ax_proc = axes[1, i]
            im_proc = ax_proc.imshow(proc_subset[i], cmap='gray', interpolation='nearest')
            ax_proc.set_title(f'Processed {labels_subset[i]}', fontsize=10)
            ax_proc.set_xticks([])
            ax_proc.set_yticks([])
            
            # Add borders
            rect_orig = patches.Rectangle((0, 0), 28, 28, linewidth=1, 
                                        edgecolor='black', facecolor='none')
            rect_proc = patches.Rectangle((0, 0), 11, 11, linewidth=1, 
                                        edgecolor='black', facecolor='none')
            ax_orig.add_patch(rect_orig)
            ax_proc.add_patch(rect_proc)
        else:
            # Empty subplots
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            axes[0, i].set_facecolor('lightgray')
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
            axes[1, i].set_facecolor('lightgray')
    
    # Add row labels
    axes[0, 0].set_ylabel('Original (28×28)', fontsize=12, rotation=90)
    axes[1, 0].set_ylabel('Processed (11×11)', fontsize=12, rotation=90)
    
    plt.tight_layout()
    return fig

def main():
    """
    Main function to create MNIST atlases.
    """
    # Load processed data
    features, labels = load_processed_mnist()
    
    # Reshape to images
    processed_images = reshape_to_image(features)
    
    print(f"Processed image shape: {processed_images.shape}")
    print(f"Sample label: {labels[0]}")
    
    # Create the main atlas
    fig1 = create_atlas(processed_images, labels)
    
    # Save the atlas
    output_file = 'mnist_atlas_9x9.png'
    fig1.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Atlas saved as {output_file}")
    
    # Try to create comparison with original images if available
    try:
        from sklearn.datasets import fetch_openml
        print("\nLoading original MNIST for comparison...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        original_images = mnist.data.reshape(-1, 28, 28)
        
        # Create comparison atlas
        fig2 = create_comparison_atlas(original_images, processed_images, labels)
        
        # Save comparison
        comparison_file = 'mnist_comparison_atlas.png'
        fig2.savefig(comparison_file, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison atlas saved as {comparison_file}")
        
    except Exception as e:
        print(f"Could not create comparison atlas: {e}")
    
    # Show the plots
    plt.show()
    
    print("\n✓ Atlas creation complete!")

if __name__ == "__main__":
    main() 