# -*- coding: utf-8 -*-
"""
Spyder Editor

DataToddler blog post 
Why is overfitting bad?
"""

import numpy as np
import matplotlib.pyplot as plt

N_sample = 30
poly_coeff = [0.2,1,-1,0] # hidden true polynomial coefficients
noise = 0.2 # measurement noise

#%%
np.random.seed(20180628)
x_train = np.random.uniform(-2,2,N_sample)
y_train = np.polyval(poly_coeff, x_train) + np.random.randn(N_sample,)*noise

fit_poly_order = 2
poly_coeff_fit = np.polyfit(x_train, y_train, fit_poly_order)


x_plot = np.sort(np.concatenate((np.arange(-2.2,2.2,0.01),x_train)))
y_plot = np.polyval(poly_coeff, x_plot)
y_plot_fit = np.polyval(poly_coeff_fit, x_plot)
                
plt.plot(x_plot, y_plot, 'c-', label='Ground truth')
plt.plot(x_train, y_train, '.', label='Training set')
plt.plot(x_plot, y_plot_fit, 'r-', label=str(fit_poly_order)+'nd order poly fit')
plt.xlabel('Independent variable $x$', fontsize=15)
plt.ylabel('Dependent variable $y$', fontsize=15)
plt.axis([-2.2,2.2,-1,5])
plt.legend()
plt.tight_layout()

#%%
N_validation = 20

train_std_errs = []
val_std_errs = []
for fit_poly_order in range(1,20):
    train_std_err = []
    val_std_err = []
    for repeat in range(200):
                
        x_train = np.random.uniform(-2,2,N_sample)
        y_train = np.polyval(poly_coeff, x_train) + np.random.randn(N_sample,)*noise
        
        poly_coeff_fit = np.polyfit(x_train, y_train, fit_poly_order)

        y_train_pred = np.polyval(poly_coeff_fit, x_train)
        train_std_err.append(np.std(y_train-y_train_pred))
        
        x_valid = np.random.uniform(-2,2,N_validation)
        y_valid = np.polyval(poly_coeff, x_valid) + np.random.randn(N_validation,)*noise
        y_valid_pred = np.polyval(poly_coeff_fit, x_valid)
        
        val_std_err.append(np.std(y_valid-y_valid_pred))
    val_std_errs.append(np.array(val_std_err))
    train_std_errs.append(np.array(train_std_err))
    
plt.figure()

plt.boxplot(train_std_errs,positions=np.arange(len(train_std_errs))-0.15, widths =0.2,flierprops=dict(marker='.'), whis = 1.5, patch_artist=False)
plt.boxplot(val_std_errs,positions=np.arange(len(val_std_errs))+0.15, widths =0.2,flierprops=dict(marker='.'), whis = 1.5, patch_artist=True)

plt.xticks(np.arange(len(val_std_errs)), np.arange(len(val_std_errs))+1)
plt.xlabel('Fitting polynomial order', fontsize=15)
plt.ylabel('Validation error (standard dev)', fontsize=15)
plt.plot([0,len(val_std_errs)], [noise, noise], 'r--')
plt.gca().set_yscale('log')
plt.gca().set_ylim([0.05,10])

plt.tight_layout()