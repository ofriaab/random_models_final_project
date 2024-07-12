import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Load the distribution parameters from the text file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'distribution_params.txt')

with open(file_path, 'r') as f:
    shape, loc, scale = map(float, f.read().split())
    print(f'shape:{shape}')
    print(f'loc:{loc}')
    print(f'scale:{scale}')

def generate_random_samples(shape, loc, scale, size=7000):
    # Generate random samples from the fitted gamma distribution
    return stats.gamma.rvs(shape, loc, scale, size=size)

# Generate random samples based on the fitted distribution
random_samples = generate_random_samples(shape, loc, scale, size=1000)

# Plotting the generated random samples
plt.figure(figsize=(10, 6))
plt.hist(random_samples, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Running Time')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Generated Random Running Times')
plt.grid(True)
plt.show()

print(f'Generated mean: {np.mean(random_samples)}, Generated std: {np.std(random_samples)}')
