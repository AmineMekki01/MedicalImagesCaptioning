import matplotlib.pyplot as plt
import pandas as pd

def plot_metric(metrics : pd.DataFrame , metric_name : str) -> None:
    """ Plot the metric.
    
    Args:
        metrics (pd.DataFrame) : metrics
        metric_name (str) : metric name
    
    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(metrics[f'train_{metric_name}'],color='red',label=f'train {metric_name}')
    plt.plot(metrics[f'val_{metric_name}'],color='orange',label=f'valid {metric_name}')
    plt.title(f'{metric_name}, lower=better')
    plt.legend()
    plt.show()