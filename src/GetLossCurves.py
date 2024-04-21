import mlflow
import matplotlib.pyplot as plt

# Fetch the logged metrics
max_epochs = 10
client = mlflow.tracking.MlflowClient()
run_id = mlflow.active_run().info.run_id
epochs = range(1, max_epochs + 1)

# Plot training loss
train_loss = client.get_metric_history(run_id, "train_loss")
train_cos_loss = client.get_metric_history(run_id, "train_cos_loss")
train_recon_loss = client.get_metric_history(run_id, "train_recon")

plt.figure()
plt.plot(epochs, [metric.value for metric in train_loss], label="Training Loss")
plt.plot(epochs, [metric.value for metric in train_cos_loss], label="Training Cosine Loss")
plt.plot(epochs, [metric.value for metric in train_recon_loss], label="Training Reconstruction Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("train_loss_graph.png")
mlflow.log_artifact("train_loss_graph.png")

# Plot validation loss
val_loss = client.get_metric_history(run_id, "val_loss")
val_cos_loss = client.get_metric_history(run_id, "val_cos_loss")
val_recon_loss = client.get_metric_history(run_id, "val_recon")

plt.figure()
plt.plot(epochs, [metric.value for metric in val_loss], label="Validation Loss")
plt.plot(epochs, [metric.value for metric in val_cos_loss], label="Validation Cosine Loss")
plt.plot(epochs, [metric.value for metric in val_recon_loss], label="Validation Reconstruction Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss per Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("val_loss_graph.png")
mlflow.log_artifact("val_loss_graph.png")