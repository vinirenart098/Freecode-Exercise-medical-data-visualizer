import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# Normalize data
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using pd.melt
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()

    # Rename the 'size' column to 'total'
    df_cat = df_cat.rename(columns={'size': 'total'})

    # Draw the catplot with sns.catplot()
    g = sns.catplot(data=df_cat, x='variable', y='total', hue='value', kind='bar', col='cardio')
    
    # Set labels
    g.set_axis_labels("variable", "total")
    g.set_xticklabels(rotation=45)
    g.fig.subplots_adjust(wspace=0.2)
    
    # Save the figure
    fig = g.fig
    fig.savefig('catplot.png')
    return fig

# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(corr)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the heatmap with sns.heatmap()
    sns.heatmap(corr, annot=True, fmt=".1f", linewidths=0.5, mask=mask, vmax=0.24, center=0, square=True, cbar_kws={"shrink": 0.75})

    # Save the figure
    fig.savefig('heatmap.png')
    return fig

# Call the functions
draw_cat_plot()
draw_heat_map()
