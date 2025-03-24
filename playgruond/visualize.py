import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the stats.log file.
df = pd.read_csv('stats.log')

# Group by 'opt_facotr' and calculate the mean of the other columns.
df_avg = df.groupby('opt_facotr', as_index=False).mean()

# Plot the specified columns against 'opt_facotr'.
columns_to_plot = ['blk_in_count', 'swap_in_count', 'in_spd', 'out_spd']
for column in columns_to_plot:
    plt.figure()
    plt.plot(df_avg['opt_facotr'], df_avg[column], marker='o')
    plt.title(f'{column} vs opt_facotr')
    plt.xlabel('opt_facotr')
    plt.ylabel(column)
    plt.grid(True)
    plt.savefig(f'{column}_vs_opt_facotr.png')