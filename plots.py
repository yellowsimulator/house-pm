import torch
from torchviz import make_dot
import matplotlib.pyplot as plt
from models import HousePredictorWithTransformerAttention


def plot_model_architecture() -> None:
    """
    Plot the model architecture and save it as a PNG file.
    """
    model = HousePredictorWithTransformerAttention()
    x = torch.rand(1, 10)
    location = torch.tensor([0])

    reg_output, class_output = model(x, location)
    dot = make_dot((reg_output, class_output), params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render(filename='images/model_architecture')


def plot_predictions(predicted_maintenance, real_maintenance, predicted_condition, real_condition):
    """
    Plot the predictions vs real values.
    
    Parameters:
    - predicted_maintenance: Predicted maintenance costs.
    - real_maintenance: Real maintenance costs.
    - predicted_condition: Predicted house conditions.
    - real_condition: Real house conditions.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot for maintenance cost
    axes[0].scatter(range(len(real_maintenance)), real_maintenance, alpha=0.6, edgecolors="w", linewidth=0.5, s=100, label="Real")
    axes[0].scatter(range(len(predicted_maintenance)), predicted_maintenance, alpha=0.6, edgecolors="w", linewidth=0.5, s=100, label="Predicted")
    axes[0].set_title("Predicted vs Real Maintenance Costs")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Maintenance Cost")
    axes[0].legend()
    
    # Plot for house condition
    axes[1].scatter(range(len(real_condition)), real_condition, alpha=0.6, edgecolors="w", linewidth=0.5, s=100, label="Real")
    axes[1].scatter(range(len(predicted_condition)), predicted_condition, alpha=0.6, edgecolors="w", linewidth=0.5, s=100, label="Predicted")
    axes[1].set_title("Predicted vs Real House Conditions")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("House Condition")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# Example usage (you'll need to have the real values for this to work):
# plot_predictions(maintenance_preds, real_maintenance_values, condition_preds, real_condition_values)
