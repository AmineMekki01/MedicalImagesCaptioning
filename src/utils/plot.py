import matplotlib.pyplot as plt

def plot_metric(metrics, metric_name):
    plt.plot(metrics[f'train_{metric_name}'],color='red',label=f'train {metric_name}')
    plt.plot(metrics[f'val_{metric_name}'],color='orange',label=f'valid {metric_name}')
    plt.title(f'{metric_name}, lower=better')
    plt.legend()
    plt.show()