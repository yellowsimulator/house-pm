import numpy as np
import pandas as pd
import torch

from preprocessing import test_loader
from models import HousePredictorWithTransformerAttention


def predict(model, test_loader) -> tuple:
    """
    Predict maintenance costs and house conditions on test data.
    
    Parameters:
        model: The trained model.
        test_loader: DataLoader for the test data.
    
    Returns:
        maintenance_predictions: Predicted maintenance costs.
        condition_predictions: Predicted house conditions.
    """
    model.eval()  # Set model to evaluation mode
    
    maintenance_predictions = []
    condition_predictions = []
    
    with torch.no_grad():
        for features, location in test_loader:
            outputs_maint, outputs_cond = model(features, location.squeeze())
            
            maintenance_predictions.append(outputs_maint.squeeze().cpu().numpy())
            condition_predictions.append(torch.argmax(outputs_cond, dim=1).cpu().numpy())
    
    # Convert list of arrays to single numpy array
    maintenance_predictions = np.concatenate(maintenance_predictions, axis=0)
    condition_predictions = np.concatenate(condition_predictions, axis=0)
    
    return maintenance_predictions, condition_predictions

if __name__ == '__main__':
    model = HousePredictorWithTransformerAttention(input_dim=10)
    maintenance_cost_preds, condition_preds = predict(model, test_loader)
