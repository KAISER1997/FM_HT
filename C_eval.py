import torch
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from marginalTailAdaptiveFlow.utils.flows import experiment,compute_arealoglog
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_wasserstein_distances(generate_data: np.ndarray, test_data: np.ndarray):
    """
    Computes and plots Wasserstein distances for each feature between generated and test data.

    Parameters:
    - generate_data: np.ndarray of shape (N, D)
    - test_data: np.ndarray of shape (N, D)
    """
    assert generate_data.shape[1] == test_data.shape[1], "Feature dimensions must match"
    num_features = generate_data.shape[1]
    
    distances = [
        wasserstein_distance(test_data[:, i], generate_data[:, i])
        for i in range(num_features)
    ]
    

    return distances  


def compute_kurtosis_ratio(
    generated_data: torch.Tensor, 
    test_data: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:

    generated_np = generated_data.flatten().cpu().numpy()
    test_data_np = test_data.flatten().cpu().numpy()


    k_sim = stats.kurtosis(generated_np, fisher=True, bias=True)
    k_data = stats.kurtosis(test_data_np, fisher=True, bias=True)
    print(k_sim,k_data)

    kr = abs(1 - (k_sim / (k_data + eps)))
    
    return torch.tensor(kr, dtype=torch.float32)

def compute_skewness_ratio(
    generated_data: torch.Tensor, 
    test_data: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:


    generated_np = generated_data.flatten().cpu().numpy()
    test_data_np = test_data.flatten().cpu().numpy()
    s_sim = stats.skew(generated_np, bias=True)
    s_data = stats.skew(test_data_np, bias=True)
    sr = abs(1 - (s_sim / (s_data + eps)))

    return torch.tensor(sr, dtype=torch.float32)

def compute_loglogarea(dimension,num_heavy,generated_data,test_data):
    area = []
    for j in range(dimension):
        print("D-",j)
        marginal_area = compute_arealoglog(test_data[:, j], generated_data[:, j])
        area.append(np.round(marginal_area, 5))

    
    area_heavy = np.mean(area[-num_heavy:])

    print(f"area_heavy {area_heavy}")
    if dimension!=num_heavy:
        area_light = np.mean(area[:-num_heavy])
        print(f"area_light {area_light}")


def compute_tvar(dimension,num_heavy,generated_data,test_data):
    tvar_dif = []
    num_samples=generated_data.shape[0]
    for component in range(dimension):
        sorted_abs_samps_synth = np.sort(np.abs(generated_data[:, component]))
        sorted_abs_data_test = np.sort(np.abs(test_data[:, component]))
        alpha = 0.95
        tvar_gen = 1 / (1 - alpha) * np.mean(sorted_abs_samps_synth[int(alpha * num_samples):])
        tvar_test = 1 / (1 - alpha) * np.mean(sorted_abs_data_test[int(alpha * len(sorted_abs_data_test)):])
        print(tvar_gen,tvar_test)
        tvar_dif.append(np.abs(tvar_test - tvar_gen))

    tvar_heavy = np.mean(tvar_dif[-num_heavy:])
    print(f"tvar_heavy {tvar_heavy}")
    if dimension!=num_heavy:
        tvar_light = np.mean(tvar_dif[:-num_heavy])
        print(f"tvar_light {tvar_light}")



def compute_all_metrics(generated_data,full_data_test,dimension,num_heavy):
    print("KURTOSIS RATIO-")
    print(compute_kurtosis_ratio(torch.tensor(generated_data),torch.tensor(full_data_test)))
    print("SKEWNESS RATIO")
    print(compute_skewness_ratio(torch.tensor(generated_data),torch.tensor(full_data_test)))
    print('loglogarea')
    compute_loglogarea(dimension,num_heavy,torch.tensor(generated_data),torch.tensor(full_data_test))
    print('tvar')
    compute_tvar(dimension,num_heavy,torch.tensor(generated_data),torch.tensor(full_data_test))
    print("wasserstein Distance")
    wasdist_list=plot_wasserstein_distances(generated_data,torch.tensor(full_data_test))
    print(wasdist_list)
    print(np.mean(wasdist_list))    



def plot_kde_subplots(test_data, gen_data, n_cols=3, figsize=(15, 4),name,approach, suptitle=None):
    """
    Plots KDE comparisons between test and generated data for each dimension.

    Parameters:
    - test_data: np.ndarray of shape (batch, dim)
    - gen_data: np.ndarray of shape (batch, dim)
    - n_cols: Number of columns in subplot grid
    - figsize: Tuple (width, height) for each subplot row
    - suptitle: Optional super title for the figure
    """
    assert test_data.shape == gen_data.shape, "test_data and gen_data must have the same shape"
    
    batch, dim = test_data.shape
    n_rows = (dim + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1]*n_rows))
    axes = axes.flatten()

    for i in range(dim):
        ax = axes[i]
        sns.kdeplot(np.clip(test_data[:, i],-200,200), ax=ax, label='Test', color='blue')
        sns.kdeplot(np.clip(gen_data[:, i],-200,200), ax=ax, label='Generated', color='orange')
        ax.set_title(f'Dimension {i}')
        ax.legend()

    # Remove unused axes
    for j in range(dim, len(axes)):
        fig.delaxes(axes[j])

    if suptitle:
        plt.suptitle(suptitle, fontsize=16)
        plt.subplots_adjust(top=0.92)

    plt.tight_layout()
    plt.savefig(name+'_'+str(approach)+"kde_comparison.png", dpi=300, bbox_inches='tight')   

  
    plt.show()




