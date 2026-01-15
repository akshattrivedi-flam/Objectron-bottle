import modal
import json
import matplotlib.pyplot as plt
import os

# Define the image and app
image = modal.Image.debian_slim().pip_install("matplotlib", "numpy")
app = modal.App("objectron-results-viewer", image=image)

# Define the same volume we used for training
volume = modal.NetworkFileSystem.from_name("objectron-data")

@app.function(network_file_systems={"/data": volume})
def plot_training_results():
    history_path = "/data/Objectron-bottle/Objectron/results/training_history.json"
    
    if not os.path.exists(history_path):
        print(f"❌ Could not find training history at {history_path}")
        # List what is available
        print("Available files in /data/Objectron-bottle/Objectron/results/:")
        if os.path.exists("/data/Objectron-bottle/Objectron/results/"):
            print(os.listdir("/data/Objectron-bottle/Objectron/results/"))
        return

    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Objectron Bottle Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the volume so we can download it
    plot_path = "/data/training_plot.png"
    plt.savefig(plot_path)
    print(f"✅ Training plot saved to cloud volume: {plot_path}")
    
    return plot_path

@app.local_entrypoint()
def main():
    print("📈 Fetching training results from Modal...")
    plot_training_results.remote()
    
    # Now download the results to local machine
    print("📥 Downloading results to your local 'results' folder...")
    os.makedirs("results", exist_ok=True)
    
    # Use subprocess to call modal volume get
    import subprocess
    subprocess.run(["modal", "volume", "get", "objectron-data", "/data/training_plot.png", "results/training_plot.png"])
    subprocess.run(["modal", "volume", "get", "objectron-data", "/data/Objectron-bottle/Objectron/results/training_history.json", "results/training_history.json"])
    
    print("\n✨ Done! You can find the results here:")
    print("1. Plot: results/training_plot.png")
    print("2. History: results/training_history.json")
