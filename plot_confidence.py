import matplotlib.pyplot as plt

def plot_confidence(stats):
    """
    Plot cross validation results with confidence intervals.
    
    Args:
        stats (dict): Dictionary containing:
            - fold_results: List of dicts with 'val_accuracy' and 'val_loss'
            - mean_val_accuracy: Mean validation accuracy
            - std_val_accuracy: Standard deviation of validation accuracy
            - mean_val_loss: Mean validation loss
            - std_val_loss: Standard deviation of validation loss
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy results
    plt.subplot(1, 2, 1)
    fold_numbers = range(1, len(stats["fold_results"]) + 1)
    accuracies = [r["val_accuracy"] for r in stats["fold_results"]]
    plt.plot(fold_numbers, accuracies, "bo-", label="Fold Accuracy")
    
    mean_acc = stats["mean_val_accuracy"]
    std_acc = stats["std_val_accuracy"]
    plt.axhline(y=mean_acc, color="r", linestyle="--", label="Mean Accuracy")
    plt.fill_between(
        fold_numbers,
        mean_acc - std_acc,
        mean_acc + std_acc,
        color="r",
        alpha=0.2,
        label="±1 std dev",
    )
    plt.xlabel("Fold")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Cross Validation Accuracy")
    plt.legend()
    plt.grid(True)

    # Plot loss results 
    plt.subplot(1, 2, 2)
    losses = [r["val_loss"] for r in stats["fold_results"]]
    plt.plot(fold_numbers, losses, "go-", label="Fold Loss")
    
    mean_loss = stats["mean_val_loss"]
    std_loss = stats["std_val_loss"]
    plt.axhline(y=mean_loss, color="r", linestyle="--", label="Mean Loss")
    plt.fill_between(
        fold_numbers,
        mean_loss - std_loss,
        mean_loss + std_loss,
        color="r",
        alpha=0.2,
        label="±1 std dev",
    )
    plt.xlabel("Fold")
    plt.ylabel("Validation Loss")
    plt.title("Cross Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show() 