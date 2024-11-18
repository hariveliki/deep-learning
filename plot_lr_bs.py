import matplotlib.pyplot as plt
import numpy as np

def plot_lr_bs_results(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Create epochs array
    epochs = range(1, len(next(iter(results.values()))['train_loss']) + 1)
    
    # Plot training loss
    for name, history in results.items():
        ax1.plot(epochs, history['train_loss'], marker='o', label=name)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # Plot validation loss
    for name, history in results.items():
        ax2.plot(epochs, history['valid_loss'], marker='o', label=name)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()

    # Plot training accuracy
    for name, history in results.items():
        ax3.plot(epochs, history['train_accuracy'], marker='o', label=name)
    ax3.set_title('Training Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.grid(True)
    ax3.legend()

    # Plot validation accuracy
    for name, history in results.items():
        ax4.plot(epochs, history['valid_accuracy'], marker='o', label=name)
    ax4.set_title('Validation Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.grid(True)
    ax4.legend()
