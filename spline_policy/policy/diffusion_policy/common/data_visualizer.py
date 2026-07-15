import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from einops import reduce


def plot_trajectory_comparison(pred, target, num_samples_to_plot=3, title_prefix=""):
    """
    Visualizes and compares model predictions against target trajectories.

    Args:
        pred (torch.Tensor): The model's prediction tensor with shape (B, horizon, Da).
        target (torch.Tensor): The ground truth tensor with shape (B, horizon, Da).
        num_samples_to_plot (int): The number of samples to plot from the batch.
        title_prefix (str): A prefix for the main title of the plot.
    """
    # Ensure we don't try to plot more samples than available in the batch
    B, horizon, Da = pred.shape
    num_samples_to_plot = min(B, num_samples_to_plot)

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    # Iterate through the number of samples to plot
    for i in range(num_samples_to_plot):
        # Create a new figure for each sample
        # We will create Da subplots and arrange them in a grid
        num_cols = 5  # Set a maximum columns per row
        num_rows = (Da + num_cols - 1) // num_cols  # Calculate required rows (ceiling division)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4), squeeze=False)
        
        # Add a main title to the entire figure
        fig.suptitle(f'{title_prefix}Sample {i+1}/{B}: Prediction vs. Target Data', fontsize=16)

        # Iterate through each dimension of the action/state
        for j in range(Da):
            row, col = j // num_cols, j % num_cols
            ax = axes[row, col]

            # Plot the prediction (red dashed line)
            ax.plot(pred_np[i, :, j], 'r--', label='Prediction (pred)')
            
            # Plot the target (blue solid line)
            ax.plot(target_np[i, :, j], 'b-', label='Target (data)')
            
            ax.set_title(f'Action Dimension {j}')
            ax.set_xlabel('Time Step (horizon)')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        # Hide any unused subplots if Da is not a multiple of num_cols
        for j in range(Da, num_rows * num_cols):
            row, col = j // num_cols, j % num_cols
            axes[row, col].axis('off')

        # Adjust layout to make room for the suptitle and prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        plt.show()
        
def plot_trajectory(trajectory_data, title_prefix=""):
    """
    Visualizes a single agent position trajectory.

    Args:
        trajectory_data (np.ndarray): The agent's position data, 
                                      expected to have a shape of (T, D_pos).
        title_prefix (str): A prefix for the main title of the plot.
    """
    # Ensure input is a NumPy array
    if not isinstance(trajectory_data, np.ndarray):
        raise TypeError(f"Input must be a NumPy array, but got {type(trajectory_data)}")

    # Get dimensions: T = Time steps, D_pos = Position dimensions
    if trajectory_data.ndim != 2:
        raise ValueError(f"Input array must be 2D (T, D_pos), but got shape {trajectory_data.shape}")
    
    T, D_pos = trajectory_data.shape

    # Arrange subplots in a grid
    num_cols = 5  # Max 3 columns
    num_rows = (D_pos + num_cols - 1) // num_cols  # Calculate required rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4), squeeze=False)

    # Add a main title for the entire figure
    fig.suptitle(f'{title_prefix}Agent Position Trajectory (T={T}, D_pos={D_pos})', fontsize=16)

    # Iterate through each dimension of the position
    for j in range(D_pos):
        row, col = j // num_cols, j % num_cols
        ax = axes[row, col]

        # Plot the trajectory for the j-th dimension
        ax.plot(trajectory_data[:, j], 'b-', label=f'Trajectory')
        
        ax.set_title(f'Position Dimension {j}')
        ax.set_xlabel('Time Step (T)')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # Hide any unused subplots
    for j in range(D_pos, num_rows * num_cols):
        row, col = j // num_cols, j % num_cols
        axes[row, col].axis('off')

    # Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()