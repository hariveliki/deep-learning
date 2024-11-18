import matplotlib.pyplot as plt


def plot_conv_settings(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(next(iter(results.values()))["train_loss"]) + 1)

    for name, history in results.items():
        ax1.plot(epochs, history["train_loss"], marker="o", label=name)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    for name, history in results.items():
        ax2.plot(epochs, history["valid_loss"], marker="o", label=None)
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True)

    for name, history in results.items():
        ax3.plot(epochs, history["train_accuracy"], marker="o", label=None)
    ax3.set_title("Training Accuracy")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy (%)")
    ax3.grid(True)

    for name, history in results.items():
        ax4.plot(epochs, history["valid_accuracy"], marker="o", label=None)
    ax4.set_title("Validation Accuracy")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy (%)")
    ax4.grid(True)

    fig.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
    plt.tight_layout()
