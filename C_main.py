#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from data.C_data import Gen_Dataset
from C_EVALS.C_eval import compute_all_metrics,plot_kde_subplots
from C_MODELS.C_model import heavy_tail_FM,Light_tail_FM,heavy_tail_input
from C_NETS.network import MLP,MLP_TailParam,MLP_TailParam2,MLP2,BigTimeConditionalNet,TimeToVecNet,FullConnectedScoreModel,FullConnectedScoreModel_time,MLP_W_ttflayers
from torch.autograd.functional import jacobian
from torch.distributions import Independent, Normal
from C_MODELS.utils.extreme_transforms import TailAffineMarginalTransform_SeparateNetParam2   
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# In[ ]:





# In[2]:


Datasetz='studentT'
method='TTF_GRAD_LIGHT_INP'   #VANILLA, TTF_GRAD, TTF_GRAD_LIGHT_INP, TTF_FINAL_LAYER, TTF_FINAL_LAYER_dTTFInverse_dz,Vanilla_studentT
if method=='VANILLA'or method=='TTF_FINAL_LAYER' or method=='TTF_FINAL_LAYER_dTTFInverse_dz' or method=='Vanilla_studentT':  
    approach=1
else:
    approach=2


# In[3]:


print("METHOD-",method)
print(Datasetz)


# In[4]:


if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')
torch.manual_seed(42)
seed=42


# In[5]:


Param,Data=Gen_Dataset(Datasetz)


# In[6]:


dimension=Param['dimension']
num_heavy=Param['num_heavy']
name=Param['name']
hidden_dim=512
batch_size=4096
lr=0.0001
iterations=1600
STEPS=12000


# In[7]:


if method=='TTF_FINAL_LAYER':
    transform='basic'
    model=MLP_W_ttflayers(input_dim=dimension, time_dim=1, hidden_dim=hidden_dim,transform=transform).to(device)
elif method=='TTF_FINAL_LAYER_dTTFInverse_dz':
    transform='Recip_deriv_inv'
    model=MLP_W_ttflayers(input_dim=dimension, time_dim=1, hidden_dim=hidden_dim,transform=transform).to(device)
else:
    model=MLP(input_dim=dimension, time_dim=1, hidden_dim=hidden_dim).to(device)

Tail_paramNet=MLP_TailParam2(time_dim=1, hidden_dim=hidden_dim//2,output_dim=4*dimension,transform_inp=0).to(device)
noise2data=TailAffineMarginalTransform_SeparateNetParam2(dimz=dimension).to(device)  #TTF


# In[8]:


full_data_train=Data['train']
full_data_test=Data['test']
full_data_val=full_data_train


# In[9]:


plt.scatter(full_data_train[:,0],full_data_train[:,1])
plt.xlim(-60,60)
plt.ylim(-60,60)
plt.show()


# In[10]:


student_t = torch.distributions.StudentT(df=2)
class NoiseDataset(Dataset):
    def __init__(self, length, noise_shape):
        self.length = length
        self.noise_shape = noise_shape
        self.precomputed_student_t = student_t.sample((length, noise_shape[0]))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if method=='Vanilla_studentT':
            
            return self.precomputed_student_t[idx] 
        
        else:

            return torch.randn(self.noise_shape)  # standard normal noise


# In[ ]:





# In[11]:


train_dataset = TensorDataset(torch.tensor(full_data_train))
noise_shape = (full_data_train.shape[1],)  # shape of each noise sample (e.g., 100-D vector)
noise_dataset = NoiseDataset(length=len(train_dataset), noise_shape=noise_shape)
noise_loader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=True)


# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# In[12]:


def freeze_model(modelz):
    for param in modelz.parameters():
        param.requires_grad=False
def unfreeze_model(modelz):
    for param in modelz.parameters():
        param.requires_grad=True


# In[ ]:





# In[ ]:





# In[13]:


optim1 = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-3)
optim2=torch.optim.Adam(Tail_paramNet.parameters(), lr=lr,weight_decay=1e-3)#,weight_decay=1e-3)


# In[ ]:





# In[14]:


if method=='VANILLA' or  method=='TTF_FINAL_LAYER' or method=='TTF_FINAL_LAYER_dTTFInverse_dz' or method=='Vanilla_studentT':
    FM_MODEL=Light_tail_FM(model,dimension,iterations,device,STEPS)
elif method=='TTF_GRAD' or method=='TTF_GRAD_LIGHT_INP':
    FM_MODEL=heavy_tail_FM(Tail_paramNet,model,noise2data,dimension,iterations,device,STEPS,method)
 
    # FM_MODEL=heavy_tail_input(Tail_paramNet,model,noise2data,dimension,iterations,device,STEPS)


# In[ ]:


approach


# In[ ]:





# In[ ]:


for i in range(iterations):
    print("ITERATION-",i,"STEPS-",FM_MODEL.count)
    if FM_MODEL.count>=FM_MODEL.steps:
        break
    # if (i+1)%1400==0:
    #     unfreeze_model(FM_MODEL.Tail_paramNet)
    #     FM_MODEL.approach=2
    if method=='VANILLA' or method=='Vanilla_studentT' or method=='TTF_FINAL_LAYER' or method=='TTF_FINAL_LAYER_dTTFInverse_dz':
        FM_MODEL.train_epoch(optim1,train_loader,noise_loader,i)
    else:
        FM_MODEL.train_epoch(optim1,optim2,train_loader,noise_loader,i)

    # if (i+1)%1400==0:
    #     freeze_model(FM_MODEL.Tail_paramNet)
    #     FM_MODEL.approach=1
    


# In[ ]:


if method=='Vanilla_studentT':
    x_init=student_t.sample(full_data_test.shape)
else:
    x_init=torch.randn(full_data_test.shape)

generated_data,traj_path=FM_MODEL.generate(x_init.to(device))


# In[ ]:


TT = torch.linspace(0,1,10)  # sample times
TT = TT.to(device)


# In[ ]:


traj_path.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import contextlib
import os

def create_experiment_structure(datasetname: str, method: str, experiment_number: int):
    # Define the base experiment path
    base_path = f"experiment_{experiment_number}"
    
    # Define the full path
    full_path = os.path.join(base_path, datasetname, method)
    
    # Create directories as needed
    os.makedirs(full_path, exist_ok=True)
    
    print(f"Created or verified path: {full_path}")
    return(full_path)
    
def save_output_to_file(func, filename: str, *args, **kwargs):
    """
    Runs `func(*args, **kwargs)` and saves all printed output to `filename`.
    """
    with open(filename, 'w') as f:
        with contextlib.redirect_stdout(f):
            func(*args, **kwargs)

def plot_trajectories(traj_path, show_start=True, show_end=True, title="Flow Trajectories"):
    """
    Plot continuous flow trajectories from a tensor of shape (T, N, D)

    Parameters:
    - traj_path: np.ndarray of shape (T, N, D), where:
        T = time steps, N = number of trajectories, D = dimensions (2D only)
    - show_start: plot start points
    - show_end: plot end points
    - title: title for the plot
    """
    T, N, D = traj_path.shape
    assert D == 2, "Only 2D trajectories are supported."

    plt.figure(figsize=(8, 6))

    for i in range(N):
        traj = traj_path[:, i, :]  # Shape: (T, 2)
        plt.plot(traj[:, 0], traj[:, 1], linewidth=2, alpha=0.7)
        if show_start:
            plt.scatter(traj[0, 0], traj[0, 1], color='blue', label='Start' if i == 0 else "")
        if show_end:
            plt.scatter(traj[-1, 0], traj[-1, 1], color='red', label='End' if i == 0 else "")

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis('equal')
    if show_start or show_end:
        plt.legend()
    plt.show()


# In[ ]:


plot_trajectories(traj_path[:,0:1000,:])


# In[ ]:


full_path=create_experiment_structure(Datasetz, method, experiment_number=1)


# In[ ]:





# In[ ]:


v=torch.zeros(TT.shape[0],dimension).to(device)
for u in range(5000):
    oz=FM_MODEL.flow_model(torch.tensor(traj_path[:,u,:]).to(device),TT)
    v=v+torch.abs(oz)
v=v/5000

for u in range(dimension):
    plt.plot(TT.cpu(),v[:,u].cpu().detach())
plt.savefig(os.path.join(full_path,'modvel.jpeg'))


# In[ ]:


full_path


# In[ ]:


path1 = os.path.join(full_path, 'generated_data.npy')  
path2 = os.path.join(full_path, 'traj_data.npy') 
path3 = os.path.join(full_path, 'test_data.npy') 

np.save(path1, generated_data)
np.save(path2, traj_path)
np.save(path3,full_data_test)


# In[ ]:


torch.save(Tail_paramNet.state_dict(), os.path.join(full_path,'tailmodel.pth'))
torch.save(model.state_dict(),os.path.join(full_path,"model.pth"))


# In[ ]:


generated_data=generated_data

plt.figure(figsize=(6, 6))
plt.scatter(full_data_test[0:10000, 0], full_data_test[0:10000, 1], color='red', label='Real Data')
plt.scatter(generated_data[0:10000, 0], generated_data[0:10000, 1], color='blue', label='Generated Data')
plt.xlim(-90, 90)
plt.ylim(-90, 90)
plt.legend()
plt.title("Generated vs Real Data")

# Save the figure to file
plt.savefig(os.path.join(full_path,'scatter_plot.png'), dpi=300, bbox_inches='tight')


# In[ ]:


generated_data=generated_data

plt.scatter(generated_data[0:10000,0],generated_data[0:10000,1])
plt.scatter(full_data_test[0:10000,0],full_data_test[0:10000,1],color='red')

plt.xlim(-90,90)
plt.ylim(-90,90)


# In[ ]:


# param_tail=FM_MODEL.Tail_paramNet(1+torch.zeros(4096).unsqueeze(1).to(device))
# param_tail_pre=param_tail#self.Tail_paramNet(1+torch.zeros(num_samples,1).to(self.device)) #BX80
# dummy_tail_param=param_tail_pre.reshape(param_tail_pre.shape[0],4,dimension)
# const=-10
# _unc_pos_tail,_unc_neg_tail,shift,_unc_scale =dummy_tail_param[:,0,:]**2+const,dummy_tail_param[:,1,:]**2+const,dummy_tail_param[:,2,:]*0,dummy_tail_param[:,3,:]*0         
# param_tail=torch.cat([_unc_pos_tail,_unc_neg_tail,shift,_unc_scale],1)

# x_init1 = torch.randn((batch_size, dimension), dtype=torch.float32, device=device)
# x_init2=FM_MODEL.TTF(x_init1,param_tail)
# # x_init1=x_init1*(x_init1>3)
# # x_init2=x_init2*(x_init1>3)
# plot_kde_subplots(x_init1.detach().cpu(),x_init2.detach().cpu(),'name',approach)


# In[ ]:


# param_tail[0][0:4]


# In[ ]:


# torch.sum(x_init1>4),torch.sum(x_init2>4)


# In[ ]:


full_path


# In[ ]:


plot_kde_subplots(full_data_test[0:generated_data.shape[0]],generated_data,os.path.join(full_path,'kde.jpeg'),approach)


# In[ ]:


# plot_kde_subplots(full_data_test[0:generated_data.shape[0]],generated_data,'name',approach)


# In[ ]:


with open(os.path.join(full_path,'eval.txt'), 'w') as f:
    with contextlib.redirect_stdout(f):
        compute_all_metrics(generated_data[0:10000],full_data_test[0:10000],dimension,num_heavy)


# In[ ]:


# KURTOSIS RATIO-
 
# -0.981878966934683 -0.9757450175970037
# tensor(0.0063)
# SKEWNESS RATIO
# tensor(9.3467)
# loglogarea
# D- 0
 
# D- 1
# area_heavy 1.193295
# tvar
# 1109.7395324707022 1084.123001098632
# 1116.3822174072257 1091.597137451171
# tvar_heavy 25.2008056640625
# wasserstein Distance
 
# [2.2989951312207357, 1.4577777797402838]
# 1.8783864554805096
# #ttf  inp light 12000 epoch


# In[ ]:


# KURTOSIS RATIO-
# -0.986062207935158 -0.9757450175970037
# tensor(0.0106)
# SKEWNESS RATIO
# tensor(13.4852)
# loglogarea
# area_heavy 1.2178849999999999
# tvar
# 1100.6172943115225 1084.123001098632
# 1112.2029876708975 1091.597137451171
# tvar_heavy 18.550071716308594
# wasserstein Distance
# [0.9005596924624412, 2.173501701538925]
# 1.537030697000683

# ttf 12000 epoch


# In[ ]:


# KURTOSIS RATIO-
# -1.0020605736813577 -0.9757450175970037
# tensor(0.0270)
# SKEWNESS RATIO
# tensor(1.5075)
# loglogarea
# D- 0
# D- 1
# area_heavy 1.2263600000000001
# tvar
# 1121.2825012207022 1084.123001098632
# 1140.703582763671 1091.597137451171
# tvar_heavy 43.132972717285156
# wasserstein Distance
# [1.4701023753320306, 2.3802090693099394]
# 1.925155722320985
# vanilla 12000epoch


# In[ ]:


# KURTOSIS RATIO-
# -0.9915240417857571 -0.9757450175970037
# tensor(0.0162)
# SKEWNESS RATIO
# tensor(4.0863)
# loglogarea
# area_heavy 1.260215
# tvar
# 1087.9920196533194 1084.123001098632
# 1082.000198364257 1091.597137451171
# tvar_heavy 6.732978820800781
# wasserstein Distance
# [1.553469739500384, 1.548204548176682]
# 1.550837143838533
# ttflast layer 12000 epoch


# In[ ]:


# import numpy as np
# import matplotlib.pyplot as plt

# # Example data
# start_points = np.array([
#     [0.0, 0.0],
#     [1.0, 0.0],
#     [0.0, 1.0]
# ])

# paths = [
#     np.array([[0.0, 0.0], [0.2, 0.1], [0.5, 0.5], [1.0, 1.0]]),
#     np.array([[1.0, 0.0], [1.2, 0.1], [1.5, 0.5], [2.0, 1.0]]),
#     np.array([[0.0, 1.0], [0.1, 1.2], [0.5, 1.5], [1.0, 2.0]])
# ]

# # Plotting
# plt.figure(figsize=(8, 6))
# for path in paths:
#     plt.plot(path[:, 0], path[:, 1], linewidth=2)
#     plt.scatter(path[0, 0], path[0, 1], color='blue')  # start point
#     plt.scatter(path[-1, 0], path[-1, 1], color='red')  # end point

# plt.title("Flow Paths from Start Points")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.grid(True)
# plt.axis('equal')
# plt.show()

