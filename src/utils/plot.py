import matplotlib.pyplot as plt
import pandas as pd
def plot_metric(metrics : pd.DataFrame , metric_name : str):
    """
    Plots metric values stored in a pandas DataFrame.   

    Parameters  
    ----------  
    metrics : pd.DataFrame  
        Pandas DataFrame containing metric values.  
    metric_name : str   
        Name of the metric. Must be present in the DataFrame columns.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(metrics[f'train_{metric_name}'],color='red',label=f'train {metric_name}')
    plt.plot(metrics[f'val_{metric_name}'],color='orange',label=f'valid {metric_name}')
    plt.title(f'{metric_name}, lower=better')
    plt.legend()
    plt.show()