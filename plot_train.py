import matplotlib.pyplot as plt

def plot_train(results, fold=None):
    """
    Plot and save training results
    
    Args:
        results: Dictionary containing training history
        fold: Current fold number (optional)
    
    Returns:
        fig: The matplotlib figure object
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(results["train_loss"], label="Train Loss")
    plt.plot(results["valid_loss"], label="Valid Loss") 
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(results["train_accuracy"], label="Train Accuracy")
    plt.plot(results["valid_accuracy"], label="Valid Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch") 
    plt.ylabel("Accuracy (%)")
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot with fold number if provided
    filename = f"plots/fold_{fold}.png" if fold is not None else "training_plot.png"
    plt.savefig(filename)
    plt.close()
    
    return fig
