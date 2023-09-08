import torch.optim as optim
import torch
from models import HousePredictorWithTransformerAttention
from preprocessing import train_loader, val_loader
import torch.nn as nn
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

num_epochs = 60
experiment_name = f"house-predictor_{num_epochs}_epochs"

# Ensure the experiment exists
experiment_id = mlflow.create_experiment(experiment_name)

mlflow.pytorch.autolog() # Automatically log parameters and metrics for PyTorch models

training_losses = []
validation_losses = []

# Start the MLflow run
with mlflow.start_run(experiment_id=experiment_id, run_name=experiment_name) as run:
    model = HousePredictorWithTransformerAttention()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_reg = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()

    # Log hyperparameters
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("num_epochs", num_epochs)

    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0
        for features, location, target_maint, target_cond in train_loader:
            # Convert target_cond to integer and ensure labels are in range [0, 4]
            target_cond = target_cond.long().squeeze()
            if target_cond.max() > 4:
                target_cond -= 1

            optimizer.zero_grad()

            # Forward pass
            outputs_maint, outputs_cond = model(features, location.squeeze())

            # Calculate losses
            loss_maint = criterion_reg(outputs_maint.squeeze(), target_maint.squeeze())
            loss_cond = criterion_class(outputs_cond, target_cond)

            # Combine and backpropagate the losses
            loss = loss_maint + loss_cond
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()

        training_loss /= len(train_loader)
        training_losses.append(training_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, location, target_maint, target_cond in val_loader:
                target_cond = target_cond.long().squeeze()
                if target_cond.max() > 4:
                    target_cond -= 1

                outputs_maint, outputs_cond = model(features, location.squeeze())
                loss_maint = criterion_reg(outputs_maint.squeeze(), target_maint.squeeze())
                loss_cond = criterion_class(outputs_cond, target_cond)
                val_loss += loss_maint.item() + loss_cond.item()

        val_loss /= len(val_loader)
        validation_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Log metrics at the end of each epoch
        mlflow.log_metric("training_loss", training_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

    print("Training complete!")

    # Save model parameters
    torch.save(model.state_dict(), f'models/house_predictor_model_parameters_{num_epochs}_epochs.pth')

    # Plot the training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses vs. Epochs')
    plt.grid(True)

    # Save the plot to a file and log it to MLflow
    plot_filename = 'loss_plot.png'
    plt.savefig(plot_filename)
    mlflow.log_artifact(plot_filename)

    # Log the model with MLflow
    mlflow.pytorch.log_model(model, "models")





# import torch.optim as optim
# import torch
# from models import HousePredictorWithTransformerAttention
# from preprocessing import train_loader, val_loader
# import torch.nn as nn
# import mlflow
# import mlflow.pytorch

# num_epochs = 30
# experiment_name = f"house-predictor_{num_epochs}_epochs"

# experiment_id = mlflow.create_experiment(experiment_name)
# #mlflow.create_experiment(name=experiment_name)
# #mlflow.pytorch.autolog() # Automatically log parameters and metrics for PyTorch models

# # Start the MLflow run
# with mlflow.start_run(experiment_id=experiment_id, run_name=experiment_name) as run:
#     model = HousePredictorWithTransformerAttention()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     criterion_reg = nn.MSELoss()
#     criterion_class = nn.CrossEntropyLoss()

#     #Log hyperparameters
#     mlflow.log_param("optimizer", "Adam")
#     mlflow.log_param("learning_rate", 0.001)
#     mlflow.log_param("num_epochs", num_epochs)

#     for epoch in range(num_epochs):
#         model.train()
#         # Training loop
#         for features, location, target_maint, target_cond in train_loader:
#             # Convert target_cond to integer and ensure labels are in range [0, 4]
#             target_cond = target_cond.long().squeeze()
#             if target_cond.max() > 4:
#                 target_cond -= 1

#             optimizer.zero_grad()

#             # Forward pass
#             outputs_maint, outputs_cond = model(features, location.squeeze())

#             # Calculate losses
#             loss_maint = criterion_reg(outputs_maint.squeeze(), target_maint.squeeze())
#             loss_cond = criterion_class(outputs_cond, target_cond)

#             # Combine and backpropagate the losses
#             loss = loss_maint + loss_cond
#             loss.backward()
#             optimizer.step()

#         # Validation loop
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for features, location, target_maint, target_cond in val_loader:
#                 # Convert target_cond to integer and ensure labels are in range [0, 4]
#                 target_cond = target_cond.long().squeeze()
#                 if target_cond.max() > 4:
#                     target_cond -= 1

#                 outputs_maint, outputs_cond = model(features, location.squeeze())
#                 loss_maint = criterion_reg(outputs_maint.squeeze(), target_maint.squeeze())
#                 loss_cond = criterion_class(outputs_cond, target_cond)
#                 val_loss += loss_maint.item() + loss_cond.item()

#         val_loss /= len(val_loader)
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {val_loss:.4f}")

#         # Log metrics at the end of each epoch
#         mlflow.log_metric("val_loss", val_loss, step=epoch)

#     print("Training complete!")

#     # Save model parameters
#     #torch.save(model.state_dict(), f'models/house_predictor_model_parameters_{num_epochs}_epochs.pth')

#     # Log the model with MLflow
#     #mlflow.pytorch.log_model(model, "models")
