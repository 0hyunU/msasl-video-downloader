# coding: utf-8
import pickle
pickle.load(open("./train/all/01986.mp4-0.pickle",'rb')).shape
lh_indices =[15,17,19,21]
a = pickle.load(open("./train/all/01986.mp4-0.pickle",'rb'))
a[:,501:,:].shape
a[:,:-42,:].shape
a[:,-42:,:].shape
a[:,493:,:].shape
a[:,493:,:][:,lh_indices,:].shape
a[:,468:,:][:,lh_indices,:].shape
a[:,468:,:][:,lh_indices,:].mean(1).shape
a[:,493:-21,:][:,lh_indices,:].shape
a[:,493:-21,:].shape
a[:,493:-21,:].mean(1).shape
a[:,468:,:][:,lh_indices,:].mean(1).shape
a[:,468:,:][:,lh_indices,:].mean(1) - a[:,493:-21,:].mean(1)
pickle.load(open("./train/all/01986.mp4-1.pickle",'rb')).shape
a = pickle.load(open("./train/all/01986.mp4-1.pickle",'rb')).shape
a[:,468:,:][:,lh_indices,:].mean(1) - a[:,493:-21,:].mean(1)
a = pickle.load(open("./train/all/01986.mp4-1.pickle",'rb'))
a[:,468:,:][:,lh_indices,:].mean(1) - a[:,493:-21,:].mean(1)
import .draw_plot import plot_2D_keypoint_every_move
from .draw_plot import plot_2D_keypoint_every_move
from draw_plot import plot_2D_keypoint_every_move
from vid.draw_plot import plot_2D_keypoint_every_move
plot_2D_keypoint_every_move(a)
a = pickle.load(open("./train/all/01986.mp4-0.pickle",'rb'))
plot_2D_keypoint_every_move(a)
a = pickle.load(open("./train/all/01986.mp4-2.pickle",'rb'))
plot_2D_keypoint_every_move(a)
a[:,468:,:][:,lh_indices,:].mean(1) - a[:,493:-21,:].mean(1)
plot_2D_keypoint_every_move(a)
a[:,468:,:][:,lh_indices,:].mean(1) - a[:,493:,:].mean(1)
a[:,468:,:][:,lh_indices,:].mean(1) - a[:,493:-21,:].mean(1)
b = a[:,468:,:][:,lh_indices,:].mean(1) - a[:,493:-21,:].mean(1)
b[0,:]**2
b.shape
b**2
b**2.shape
(b**2).shape
(b**2).sum(axis=1).shape
(b**2).sum(axis=1)
plot_2D_keypoint_every_move(a)
plot_2D_keypoint_every_move(a)
plot_2D_keypoint_every_move(a)
b = a[:,468:,:][:,lh_indices,:].mean(1) - a[:,493:-21,:].mean(1)
a[:,468:,:][:,lh_indices,:].mean(1).shape
a[:,468:,:][:,lh_indices,:].shape
b.shape
round((b**2).sum(axis=1))
round((b**2).sum(axis=1),2)
np.around((b**2).sum(axis=1),2)
import numpy as np
np.around((b**2).sum(axis=1),2)
np.around((b**2).sum(axis=1),1)
k = np.around((b**2).sum(axis=1),1)
bool(k)
k.any()
k.astype(bool)
a[k.astype(bool),:,:].shape
a[k.astype(bool),493:-21,:]  = a[k.astype(bool),lh_indices,:].mean(1)
a[k.astype(bool),493:-21,:].shape
a[k.astype(bool),lh_indices,:].mean(1)
a[k.astype(bool),:,:].shape
a[k.astype(bool),lh_indices,:].shape
a[k.astype(bool),:,:][:,lh_indices,:].shape
a[k.astype(bool),493:-21,:]  = a[k.astype(bool),:,:][:,lh_indices,:].mean(1)
b = a[:,468:,:][:,lh_indices,:].mean(1) - a[:,493:-21,:].mean(1)
(b**2).sum(axis=1)
plot_2D_keypoint_every_move(a)
np.around((b**2).sum(axis=1),1)
a[k.astype(bool),468:,:][:,lh_indices,:].shape
a[k.astype(bool),493:-21,:]  = a[k.astype(bool),468:,:][:,lh_indices,:].mean(1)
(b**2).sum(axis=1)
b = a[:,468:,:][:,lh_indices,:].mean(1) - a[:,493:-21,:].mean(1)
(b**2).sum(axis=1)
plot_2D_keypoint_every_move(a)
(b**2).sum(axis=1)
(b**2).sum(axis=1)
np.around((b**2).sum(axis=1),1)
plot_2D_keypoint_every_move(a)
plot_2D_keypoint_every_move(a)
get_ipython().run_line_magic('save', '')
get_ipython().run_line_magic('save', 'ipython.py')
get_ipython().run_line_magic('save', 'current_session ~0/')
