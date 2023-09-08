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
        real_maintenance_costs: Actual maintenance costs.
        real_conditions: Actual house conditions.
    """
    model.eval()  # Set the model to evaluation mode
    
    maintenance_predictions = []
    condition_predictions = []
    real_maintenance_costs = []
    real_conditions = []

    with torch.no_grad():  # Disable gradient calculations
        for features, location, real_maint, real_cond in test_loader:
            outputs_maint, outputs_cond = model(features, location.squeeze())
            
            # Store the predictions
            maintenance_predictions.append(outputs_maint.squeeze().cpu().numpy())
            condition_predictions.append(torch.argmax(outputs_cond, dim=1).cpu().numpy())
            real_maintenance_costs.append(real_maint.squeeze().cpu().numpy())
            real_conditions.append(real_cond.squeeze().cpu().numpy())

    # Convert lists of arrays to single numpy arrays for ease of use
    maintenance_predictions = np.concatenate(maintenance_predictions, axis=0)
    condition_predictions = np.concatenate(condition_predictions, axis=0)
    real_maintenance_costs = np.concatenate(real_maintenance_costs, axis=0)
    real_conditions = np.concatenate(real_conditions, axis=0)

    return maintenance_predictions, condition_predictions, real_maintenance_costs, real_conditions

if __name__ == '__main__':
    # Instantiate the model
    model = HousePredictorWithTransformerAttention(input_dim=10)

    # Load trained model parameters
    model_path = f'models/house_predictor_model_parameters_30_epochs.pth'
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    # Get predictions and real values
    maintenance_cost_preds, condition_preds, real_maint_costs, real_conds = predict(model, test_loader)

    # Create a DataFrame for better visualization
    predictions_df = pd.DataFrame({
        'Predicted Maintenance Costs': maintenance_cost_preds,
        'Real Maintenance Costs': real_maint_costs,
        'Predicted House Condition': condition_preds + 1,  # Adding 1 as you did before
        'Real House Condition': real_conds + 1
    })

    # Print the conditions for a quick look, you can adjust this to print more columns if needed
    for _, row in predictions_df.iterrows():
        print(f"Predicted: {row['Predicted Maintenance Costs']}, Real: {row['Real Maintenance Costs']}")
        print(f"Predicted: {row['Predicted House Condition']}, Real: {row['Real House Condition']}")
        print("-------------------------------------------")
        print("-------------------------------------------")
