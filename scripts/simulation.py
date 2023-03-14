# %%
import torch 
import numpy as np
import matplotlib.pyplot as plt

# %%
tau_a = 25
tau_s = 15
tau_adp = 20

b_0 = 0.2
is_adp = True

# %%
def shifted_sigmoid(x):
    return (1/(1+np.exp(-x*5))-0.5)/6

plt.plot(np.arange(-2, 2, 0.01), shifted_sigmoid(np.arange(-2, 2, 0.01)))
plt.show()

# %%
def mem_update(fb, ff, soma, spk, tuft_curr, b, is_adapt=is_adp, dt=1, baseline_thre=b_0, r_m=3):
    alpha = np.exp(-dt/tau_a)
    gamma = np.exp(-dt/tau_s)
    rho = np.exp(-dt/tau_adp)

    if is_adapt:
        beta = 1.8
    else:
        beta = 0.
    
    b = rho * b + (1 - rho) * spk
    new_thre = baseline_thre + beta * b

    a = alpha * tuft_curr + fb  # saturation 
    s = gamma * soma + shifted_sigmoid(a) + ff - new_thre * spk

    spk = float((s - new_thre) > 0)

    return s, spk, a, b 


# %%
w_ff = 0.15
w_fb = 0.15

# %%
input_curr = 0.03
top_clamp = 0.0
T = 250

# %%
neuron1 = {
    'soma': [0], 
    'tuft_curr': [0], 
    'spike': [0], 
    'thre': [b_0],
    'error': [0]
}

neuron2 = {
    'soma': [0], 
    'tuft_curr': [0], 
    'spike': [0], 
    'thre': [b_0], 
    'error': [0]
}

for i in range(T): 
    soma1, spk1, tuft_curr1, b1 = mem_update(neuron2['spike'][-1] * w_fb, input_curr, neuron1['soma'][-1], neuron1['spike'][-1], neuron1['tuft_curr'][-1], neuron1['thre'][-1])

    neuron1['soma'].append(soma1)
    neuron1['spike'].append(spk1)
    neuron1['tuft_curr'].append(tuft_curr1)
    neuron1['thre'].append(b1)
    neuron1['error'].append(tuft_curr1-soma1)

    soma2, spk2, tuft_curr2, b2 = mem_update(top_clamp, neuron1['spike'][-1] * w_ff, neuron2['soma'][-1], neuron2['spike'][-1], neuron2['tuft_curr'][-1], neuron2['thre'][-1])

    neuron2['soma'].append(soma2)
    neuron2['spike'].append(spk2)
    neuron2['tuft_curr'].append(tuft_curr2)
    neuron2['thre'].append(b2)
    neuron2['error'].append(tuft_curr2-soma2)




# %%
# neuron1['spike']

# %%
# spiking 
plt.plot(np.arange(T+1), neuron1['spike'], label='neuron 1')
plt.plot(np.arange(T+1), neuron2['spike'], label='neuron 2')
plt.legend()
plt.show()

# %%
# spk, mem, adaptive thre, a
fig = plt.figure(figsize=(10, 3))
plt.plot(np.arange(T+1), neuron1['spike'], label='spk')
plt.plot(np.arange(T+1), neuron1['soma'], label='soma')
plt.plot(np.arange(T+1), neuron1['tuft_curr'], label='tuft curr')
plt.plot(np.arange(T+1), neuron1['thre'], label='adp thre')
plt.plot(np.arange(T+1), neuron1['error'], label='error')

plt.legend()
plt.show()

# %%
fig = plt.figure(figsize=(10, 3))
plt.plot(np.arange(T+1), neuron2['spike'], label='spk')
plt.plot(np.arange(T+1), neuron2['soma'], label='soma')
plt.plot(np.arange(T+1), neuron2['tuft_curr'], label='tuft curr')
plt.plot(np.arange(T+1), neuron2['thre'], label='adp thre')
plt.plot(np.arange(T+1), neuron2['error'], label='error')

plt.legend()
plt.show()
# %%
