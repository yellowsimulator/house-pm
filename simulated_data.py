import numpy as np
import pandas as pd

class SimulatedDataset:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        
    def generate(self):
        age = np.random.randint(0, 100, self.n_samples)
        material = np.random.choice([0, 1, 2], self.n_samples, p=[0.3, 0.5, 0.2]) 
        size = np.random.randint(40, 200, self.n_samples)
        tenant_behavior = np.random.randint(1, 6, self.n_samples)
        maintenance_freq = np.random.randint(0, 5, self.n_samples)
        budget_constraints = np.random.choice([0, 1], self.n_samples, p=[0.6, 0.4]) 
        prev_maintenance_quality = np.random.choice([0, 1, 2], self.n_samples, p=[0.25, 0.55, 0.2])
        bathrooms = np.random.randint(1, 5, self.n_samples)
        location = np.random.choice(['Urban', 'Suburban', 'Rural'], self.n_samples, p=[0.5, 0.3, 0.2])
        has_garage = np.random.choice([0, 1], self.n_samples, p=[0.4, 0.6])
        rooms = np.random.randint(1, 7, self.n_samples)  # Assuming between 1 and 6 rooms
        
        # Maintenance Cost relationship with variables
        maintenance_cost = (age * 15) - (material * 100) + (size * 10) + \
                           (tenant_behavior * 50) - (maintenance_freq * 70) + \
                           (budget_constraints * 300) - (prev_maintenance_quality * 80) + \
                           (bathrooms * 120) + (has_garage * 200) + (rooms * 100)  # Rooms add to the cost
        
        if location[0] == 'Urban':
            maintenance_cost += 150
        elif location[0] == 'Rural':
            maintenance_cost -= 100
        
        #maintenance_cost += np.random.normal(0, 100, self.n_samples)  # Adding noise
        # Maintenance Cost relationship with variables
        maintenance_cost = (age * 15.0) - (material * 100.0) + (size * 10.0) + \
                   (tenant_behavior * 50.0) - (maintenance_freq * 70.0) + \
                   (budget_constraints * 300.0) - (prev_maintenance_quality * 80.0) + \
                   (bathrooms * 120.0) + (has_garage * 200.0) + (rooms * 100.0)  # Rooms add to the cost

        # House Condition relationship with variables
        condition = 6 - (age // 20) - (material) + (tenant_behavior) - \
            (maintenance_freq) + (budget_constraints * 2) + (prev_maintenance_quality) - \
            (bathrooms * 0.5) + (has_garage * 0.5) - (rooms * 0.3)  # Rooms can influence condition
        if location[0] == 'Urban':
            condition -= 1
        elif location[0] == 'Rural':
            condition += 1

        condition = np.clip(condition, 1, 5)
        condition = np.round(condition).astype(int)  # Round and convert to integer

        
        data = pd.DataFrame({
            'age': age,
            'material': material,
            'size': size,
            'tenant_behavior': tenant_behavior,
            'maintenance_frequency': maintenance_freq,
            'budget_constraints': budget_constraints,
            'previous_maintenance_quality': prev_maintenance_quality,
            'bathrooms': bathrooms,
            'location': location,
            'has_garage': has_garage,
            'rooms': rooms,
            'maintenance_cost': maintenance_cost,
            'house_condition': condition
        })
        
        return data
if __name__ == '__main__':
    # Test the dataset
    dataset = SimulatedDataset(n_samples=5)
    sample_data = dataset.generate()
    print(sample_data)

