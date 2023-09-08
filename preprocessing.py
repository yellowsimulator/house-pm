import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from simulated_data import SimulatedDataset
from torch.utils.data import DataLoader

def get_data(n_samples: int) -> tuple:
    """
    Get the data and split it into train, validation, and test sets.

    Parameters:
        n_samples (int): Number of samples to generate

    Returns:
        tuple: Train, validation, and test sets
    """
    data = SimulatedDataset(n_samples=n_samples).generate()
    features = data.drop(columns=['maintenance_cost', 'house_condition'])
    target_maintenance = data['maintenance_cost']
    target_condition = data['house_condition']
    
    # Convert location to integers
    le = LabelEncoder()
    features['location'] = le.fit_transform(features['location'])
    
    # Splitting into train and temp (to later split into validation and test)
    X_train_temp, X_test, y_maint_train_temp, y_maint_test, y_cond_train_temp, y_cond_test = train_test_split(features, target_maintenance, target_condition, test_size=0.2, random_state=42)
    
    # Splitting the temp data into actual train and validation sets
    X_train, X_val, y_maint_train, y_maint_val, y_cond_train, y_cond_val = train_test_split(X_train_temp, y_maint_train_temp, y_cond_train_temp, test_size=0.25, random_state=42)
    
    return X_train, X_val, X_test, y_maint_train, y_maint_val, y_maint_test, y_cond_train, y_cond_val, y_cond_test


class HouseDataset(Dataset):
    def __init__(self, features, target_maintenance, target_condition):
        self.features = features.drop(columns=['location']).values
        self.location = features['location'].values
        self.target_maintenance = target_maintenance.values
        self.target_condition = target_condition.values
        
        # Check the dtype of the numpy arrays
        assert self.features.dtype != 'O', "Features has object dtype"
        assert self.location.dtype != 'O', "Location has object dtype"
        assert self.target_maintenance.dtype != 'O', "Target Maintenance has object dtype"
        assert self.target_condition.dtype != 'O', "Target Condition has object dtype"

    def __len__(self):
        return len(self.target_maintenance)

    def __getitem__(self, idx):
        #print(self.features[idx].shape, [self.location[idx]].shape, [self.target_maintenance[idx]].shape, self.target_condition[idx].shape)
        #def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.LongTensor([self.location[idx]]),
            torch.FloatTensor([self.target_maintenance[idx]]),
            torch.LongTensor([self.target_condition[idx]])
        )



# Convert to DataLoader
n_samples = 10000
X_train, X_val, X_test, y_maint_train, y_maint_val, y_maint_test, y_cond_train, y_cond_val, y_cond_test = get_data(n_samples=n_samples)
train_dataset = HouseDataset(X_train, y_maint_train, y_cond_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = HouseDataset(X_val, y_maint_val, y_cond_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = HouseDataset(X_test, y_maint_test, y_cond_test)
