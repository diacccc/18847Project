import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('results.csv')

# Create a figure
plt.figure(figsize=(20, 12))

# Plot GFLOPS for each implementation
implementations = df['Implementation'].unique()
markers = ['o', 's', '^', 'D']  # Different markers for each implementation
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Different colors

for i, impl in enumerate(implementations):
    data = df[df['Implementation'] == impl]
    data = data.sort_values(by='M')  # Sort by matrix size
    plt.plot(data['M'], data['GFLOPS'], marker=markers[i], 
             label=impl, color=colors[i], linewidth=2)

# Add labels and title
plt.xlabel('Matrix Size (M=N=K)')
plt.ylabel('GFLOPS')
plt.title('GEMM Performance')

# Add legend
plt.legend()

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Set x-ticks to match matrix sizes
plt.xticks(sorted(df['M'].unique()))

# Save the plot
plt.savefig('gemm_gflops.png')

# Show the plot
plt.show()