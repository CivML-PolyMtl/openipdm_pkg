""" openIPDM Light ########################################################################################################

    OpenIPDM Light is an online version for the software openIPDM.
    Developed by: Zachary Hamida
    Email: zac.hamida@gmail.com
    Webpage: https://zachamida.github.io

    Material Types for each structural category:

    mat_beam = ("N.A.",
    'Weathering Steel',
    'Regular steel',
    'Aluminium',
    'Wood',
    'High-Performance Concrete',
    'Pre-stressed concrete',
    'Regular concrete')

    mat_frontwall = ('Regular steel',
    'Wood',
    'Wood covered with concrete',
    'High-Performance Concrete',
    'Regular concrete',
    'Masonry')

    mat_slab = ('N.A.',
    'Steel grating',
    'Slab with transverse prestressing',
    'High-performance concrete slab',
    'Latex concrete slab',
    'Regular concrete slab',
    'Intermittent slab with transverse prestressing',
    'Orthotropic slab',
    'Cross-laminated timber panel',
    'Corrugated sheet/plate with concrete',
    'Decking - Anti-deer crossing',
    'Decking - Railway bridge',
    'Aluminum decking',
    'Wood decking',
    'Glued-laminated timber decking',
    'Sandwich Panel System (SPS)')

    mat_gaurdrail = ('Regular steel',
    'Wood',
    'Wood / Steel',
    'Regular concrete')

    mat_wingwall = ('N.A.',
    'Regular steel',
    'Wood',
    'Wood covered with concrete',
    'High-Performance Concrete',
    'Regular concrete',
    'Gabions',
    'Masonry')

    mat_pave = ('Wood',
    'Concrete slab',
    'Asphalt pavement ',
    'Asbestos-containing asphalt')

"""
import numpy as np
import math as mt
import copy
import scipy.linalg as sp
import scipy.io as sio
import scipy.stats as stats
import scipy.special as sc
import scipy.interpolate as interpolate
import altair as alt
import pandas as pd
import os
from os.path import join as pjoin



class SSM_KR:
    def __init__(self, fixed_seed=0, selected_cat = 0, selected_mat = 0, selected_age = 50):

        # pre-selected structural category
        # Beams | Front Wall | Slabs | gaurdrail  | Wing Wall | Pavement
        self.selected_cat = selected_cat
        self.fixed_start = 0
        self.starting_state = np.array([ [np.random.randint(35,90), np.random.uniform(-1.5,-0.15), -0.005], [10**2, 0.05**2, 0.001**2] ])
        self.total_years = 100           # total time steps

        # functions
        self.inv = np.linalg.pinv
        self.svd = np.linalg.svd
        # time props
        self.dt = 1                      # time step size
        

        # network props
        # Network[birdge #1(Category #1(#Elements), Category #2(#Elements)), 
        #         bridge #2(Category #1(#Elements), Category #2(#Elements))
        #           .       .   .   .   .       .   .   .       .   .   .   
        #         bridge #B(Category #1(#Elements), Category #2(#Elements))  ]

        # Beams | Front Wall | Slabs | gaurdrail  | Wing Wall | Pavement
        self.net_data = np.array([
                                [[1]],
                                ])

        # load external data
        package_dir = os.path.dirname(__file__)
        filename = os.path.join(package_dir, 'data', 'service_life.mat')
        self.CDF = sio.loadmat(filename)

        filename = os.path.join(package_dir, 'data', 'Pretrained_SSMKR.mat')
        self.pretrained_ssm = sio.loadmat(filename)
        self.RegModel = RegressionModel()

        # Environment Seed
        self.fixed_seed = fixed_seed

        # reset number of bridges, structural categories and elements
        self.num_c = [ len(listElem) for listElem in self.net_data] # structural categories
        self.num_b = self.net_data.shape[0]                         # number of bridges
        self.num_e = self.net_data                                  # number of structural elements
        self.initial_state = np.zeros([self.num_b, np.max(self.num_c), np.max(np.max(self.num_e)),3]) 

        # indicators
        self.cb = np.array(0) # current bridge
        self.cc = np.array(0) # current structural category
        self.ci = np.array(0) # current structural element
        self.ec = np.array(0) # element index tracker can be utlized in the state vector, also faciltates generateing a new inspector

        # inspection data
        self.max_cond = np.array(100)                       # dynamic max health condition u_t
        self.max_cond_original = copy.copy(self.max_cond)   # fixed max health condition u_t
        self.max_cond_decay = 0.999                         # decay factor for max health condition 
        self.min_cond = np.array(25)                        # fixed min health condition 

        self.y = np.nan * np.zeros([self.num_c[self.cb], np.max(self.num_e[self.cb,:, 0])]) # inspection data y

        self.inspection_frq = np.random.randint(3,4,self.num_b)                             # inspection frequency 
        self.inspector_std = np.array(range(0,6))                                         # inspector ID
        self.inspector_std = np.c_[self.inspector_std, np.array([-1, 1, -1, 1, 0, 0]), np.array([4, 4, 1.5, 1.5, 4, 1.5])]          # inspectors' error std.

        # kenimatic model
        self.F = np.array([1, 0, 0])                                                        # observation matrix
        self.A = np.array([[1, self.dt, (self.dt ** 2) / 2], [0, 1, self.dt], [0, 0, 1]])   # transition model
        sigma_w = self.pretrained_ssm['AllElemParams'][self.selected_cat][1][0,0]                           # [first index] process error std. for the kenimatic model
        self.Q = sigma_w** 2 * np.array([[(self.dt ** 5) / 20, (self.dt ** 4) / 8, (self.dt ** 3) / 6],
                                    [(self.dt ** 4) / 8, (self.dt ** 3) / 3, (self.dt ** 2) / 2],
                                    [(self.dt ** 3) / 6, (self.dt ** 2) / 2, self.dt]])  # Process error (covariance)
        self.cs = np.empty((1,6))                                   # elements
        # prior initial state    
        self.x_init = self.starting_state                           # initial state for the structural elements in the network
        self.init_var = np.zeros([self.num_b, np.max(self.num_c), self.num_e.max(),6,6]) # intiial variance for the state

        # transformation function 
        self.n_tr = 4
        self.ST = SpaceTransformation(self.n_tr,self.max_cond,self.min_cond)
        # State constraits
        self.SC = StateConstraints()

        # estimated states
        self.e_Ex = np.empty((1,6))         # elements
        self.e_Var = np.zeros([self.num_b, np.max(self.num_c), self.num_e.max(),6,6])      # elements
        self.c_Ex = np.empty((self.num_b,np.max(self.num_c),3))     # category
        self.c_Var = np.empty((self.num_b,np.max(self.num_c),3,3))  # category
        self.b_Ex = np.empty((self.num_b,3))        # bridge
        self.b_Var = np.empty((self.num_b,3,3))     # bridge
        self.net_Ex = np.empty((1,6))               # network
        self.net_Var = np.empty((1,6,6))            # network
        
        # initilizing the network
        # biased inspector -/+ biased --/++ unbiased +, unbiased ++  
        self.inspector = np.array([4,4,4,4,4,4])
        self.current_year = 0
            
        # initilizing the deterioration model
        self.Am = sp.block_diag(self.A, np.eye(3))
        self.Qm = sp.block_diag(self.Q, np.zeros(3))
        self.Fm = np.append(self.F, np.zeros([1,3]))

        # SSM model estimated parameters
        self.sigma_v = self.inspector_std

        # interventions for Beams | Front Wall | Slabs | gaurdrail  | Wing Wall | Pavement
        self.int_Ex = np.array([  [[0.5,0.169,1e-2],
                                    [8.023,0.189,1e-2],
                                    [18.117,0.179,1e-2]],
                                        [[0.1,0.169,1e-2],
                                        [18.407,0.199,1e-3],
                                        [21.28,0.495,1e-4]],
                                            [[1.894,0.532,3.1e-4],
                                            [11.358,0.4,1e-4],
                                            [20.632,0.802,0.004]],
                                                [[0.2,0.169,1e-2],
                                                [9.354,0.499,1e-4],
                                                [13.683,0.054,1e-4]], 
                                                    [[0.3,0.169,1e-2],
                                                    [8.023,0.189,1e-2],
                                                    [18.663,0.263,1e-4]],         
                                                        [[8.023,0.189,1e-2],
                                                        [20.933,0.187,2e-2],
                                                        [27.071,0.505,2e-2]] ])

        self.int_var = np.array([[[8.28e-05,2.65e-07,-5.29e-08],
                                    [2.65e-07,0.00030,1.75e-05],
                                    [-5.29e-08,1.75e-05,0.00012]],
                                                                [[0.37,0.00087,-8.52e-05],
                                                                [0.00087,0.0048,2.62e-05],
                                                                [-8.52e-05,2.62e-05,0.00013]],
                                                                                        [[0.18,0.00027,-0.0017],
                                                                                        [0.00027,0.00029,1.45e-05],
                                                                                        [-0.0018,1.45e-05,0.00052]]])
        # interventions true state    
        self.int_true_var = np.array([[10**-8, 0.05**2, 10**-10],
                                    [2**2, 0.1**2, 10**-8],
                                    [4**2, 0.15**2, 10**-8]])
        self.int_true = np.array([ [[0.5, 0.2, 1e-2],
                                        [7.5, 0.3, 1e-2],
                                        [18.75, 0.4, 1e-2]],
                                                [[0.1,0.2,1e-2],
                                                [19,0.2,1e-3],
                                                [20.5,0.4,1e-4]],
                                                    [[1,0.5,3.1e-4],
                                                    [12,0.4,1e-4],
                                                    [20,0.8,0.004]],
                                                        [[0.25,0.169,1e-2],
                                                        [9.0,0.5,1e-4],
                                                        [14.0,0.1,1e-4]],
                                                            [[0.25,0.18,1e-2],
                                                            [8,0.17,1e-2],
                                                            [17,0.27,1e-4]],
                                                                [[8,0.19,1e-2],
                                                                [20,0.18,2e-2],
                                                                [28,0.50,2e-2]]       
                                                                                ])
        self.int_Q = np.square(np.array([[  [1.39,0,0,0,0,0],
                                            [0,0.01,0,0,0,0],
                                            [0,0,0.045,0,0,0],
                                            [0,0,0,1.39,0,0],
                                            [0,0,0,0,0.01,0],
                                            [0,0,0,0,0,0.045]],
                                                            [[3.533,0,0,0,0,0],
                                                            [0,0.0747,0,0,0,0],
                                                            [0,0,0.047,0,0,0],
                                                            [0,0,0,3.533,0,0],
                                                            [0,0,0,0,0.0747,0],
                                                            [0,0,0,0,0,0.047]],
                                                                            [[3.768,0,0,0,0,0],
                                                                            [0,0.0227,0,0,0,0],
                                                                            [0,0,0.0499,0,0,0],
                                                                            [0,0,0,3.768,0,0],
                                                                            [0,0,0,0,0.0227,0],
                                                                            [0,0,0,0,0,0.0499]]]))
        

        # RL model
        # actions on elements
        self.actions = np.array([0,1,2,3,4])
        self.actionCardinality = self.actions.shape[0]

        # actions tracking
        self.act_timer = 0     # tracking the frequency of the policy's actions
        self.actions_count = np.zeros([self.num_b, np.max(self.num_c), self.num_e.max(), self.actionCardinality])
        self.actions_hist = self.total_years * np.ones([self.num_b, np.max(self.num_c), self.num_e.max(), self.actionCardinality])

        # metrics related to the agent's actions
        self.cat_act_ratio = np.zeros((self.actionCardinality,np.max(self.num_c)))
        self.act_timer_on = 1

        # Rewards
        self.shutdown_cond = 35
        self.shutdown_speed = -2
        self.functional_cond = 70
        # interventions for Beams | Front Wall | Slabs | gaurdrail  | Wing Wall | Pavement
        # [ [45, -1.7], [50, -1.7], [50, -1.8], [30, -1.8], [45, -1.8], [50, -1.5] ]
        critical_state = [ [55, -1.5], [55, -1.5], [55, -1.5], [45, -1.8], [45, -1.8], [45, -1.8] ]
        self.element_critical_cond = critical_state[self.selected_cat][0] #
        self.element_critical_speed = critical_state[self.selected_cat][1] #

        # interventions for Beams | Front Wall | Slabs | gaurdrail  | Wing Wall | Pavement
        # Material b12 
        material_ind = selected_mat#[8, 5, 1, 3, 6, 3]
        self.struc_attributes = [[material_ind, selected_age, 48.7664]]  #[[material_ind[self.selected_cat], 50, 48.7664]]        # bridge attributes [Material, Age, Lattitude] b12: [[5, 50, 48.7664]] 

        # decaying  factors
        self.alpha1 = 1 # condition
        self.alpha2 = 1 # speed

        # plotting
        self.color_ex = 'bo'
        self.color_std = 'b'
        self.plt_y = []
        self.plt_true_c = []
        self.plt_true_s = []
        self.plt_Ex = []
        self.plt_var = []
        self.plt_Ex_dot = []
        self.plt_var_dot = []
        self.plt_goal = []
        self.plt_R = []
        self.plt_t = []


    def reset(self, y= [], total_years=[], inspector_std=[], inspector=[]):
        if y.__len__() > 0:
            self.y = y
            self.inspector = inspector
            self.inspector_std = inspector_std
            self.total_years = total_years
        # initiation
        self.get_initial_state()
        self.initial_run()
        # analyses level: network, bridge, category, element
        action = 0
        _, _, _, observation = self.elem_action(action)
        return observation

    def get_initial_state(self):
        for i in range(self.num_b):
            for j in range(self.num_c[i]):
                for k in range(0,self.num_e[i,j,0]):
                    Var_w0 = self.pretrained_ssm['AllElemParams'][j][3][0,0][2][0,0]
                    init_ex = self.pretrained_ssm['AllElemParams'][j][3][0,0][0]
                    init_var = self.pretrained_ssm['AllElemParams'][j][3][0,0][1]
                    KernelType = self.pretrained_ssm['AllElemParams'][j][3][0,0][3]
                    Kernel_l = self.pretrained_ssm['AllElemParams'][j][3][0,0][5][0]
                    X_ControlPoints = self.pretrained_ssm['AllElemParams'][j][3][0,0][4]
                    sample_state = np.random.multivariate_normal(self.x_init[0,:],np.diag(self.x_init[1,:]))
                    if ~np.isnan(self.y[0][1]):
                        sample_state[0] = self.y[0][1]
                    self.struc_attributes[self.cb].append(sample_state[0])
                    Kr = 1
                    for im in range(len(KernelType)):
                        Krv=self.RegModel.Kernel_Function(self.struc_attributes[self.cb][im],X_ControlPoints[:,im],Kernel_l[im],KernelType[im,0][0])
                        Kr=np.multiply(Kr,Krv)
                    AKr=np.divide(Kr,np.sum(Kr,1))
                    det_speed_Ex = AKr@init_ex
                    det_speed_Var = AKr@init_var@np.transpose(AKr) + Var_w0**2
                    sample_state[1] = det_speed_Ex
                    self.initial_state[i,j,k,:] = sample_state
                    self.init_var[i,j,k,0,0] = self.pretrained_ssm['AllElemParams'][j][1][0][2]**2
                    self.init_var[i,j,k,1,1] = det_speed_Var[0][0]
                    self.init_var[i,j,k,2,2] = self.pretrained_ssm['AllElemParams'][j][1][0][4]**2
                    self.init_var[i,j,k,3:6,3:6] = np.eye(3,3)

    def initial_run(self):
        self.cs = self.initial_state
        self.e_Ex = np.concatenate([np.zeros(self.cs.shape), np.zeros(self.cs.shape)],axis = 3)
        self.e_Ex[:,:,:,0:3] = self.initial_state[0,0,0,0:3]
        self.e_Ex[self.e_Ex[:,:,:,1]>0, 1] = 0
        self.e_Ex[:,:,:,2] = 0
        #e_var = np.diag(np.array(self.init_var))
        #self.e_Var = np.zeros([self.num_b, np.max(self.num_c), self.num_e.max(),6,6])
        self.e_Var = self.init_var
        for i in range(self.num_b):
            self.cb = i
            for j in range(self.num_c[i]):
                self.cc = j
                for k in range(self.num_e[i,j,0]):
                    self.ci = k
                    self.estimate_step(self.e_Ex[self.cb,self.cc,self.ci,:], self.e_Var[self.cb, self.cc, self.ci, :, :], 0)
        self.ci = 0
        self.cc = 0
        self.cb = 0
        self.ec = self.num_e[self.cb,self.cc,0]

    def step(self, action): # action is a scalar
        reward = 0
        # action 
        _, _, reward, observation = self.elem_action(action)
        # done
        done = 0 

        info = []

        return observation, reward, done, info
                
    def estimate_step(self, mu, var, action):
        if action == 0 or action == 4:
            int_mu = np.zeros([3])
            int_Sigma = np.eye(3,3)
            Q_int = np.zeros([6,6])
            self.Am[0:3,3:6] = np.zeros([3,3])
        else:
            int_mu = copy.copy(self.int_Ex[self.cc][action-1])
            int_Sigma = copy.copy(np.diag(self.int_var[0][action-1]))
            Q_int = self.int_Q[action-1]
            self.Am[0:3,3:6] = np.eye(3)
        all_noise = sp.block_diag(self.Q,np.zeros([3,3])) + Q_int
        if mu.size == mu.squeeze().shape[0]:
            mu = mu[np.newaxis,:]
            var = var[np.newaxis,:,:]
            # check for actions timeline
            self.actions_hist[self.cb, self.cc, self.ci, :] += 1
            if np.any(self.actions_hist[self.cb, self.cc, self.ci, :] > self.total_years) :
                check_max = np.where(self.actions_hist[self.cb, self.cc, self.ci, :] > self.total_years)
                self.actions_hist[self.cb, self.cc, self.ci, check_max] = self.total_years

            if action != 4:
                self.actions_hist[self.cb, self.cc, self.ci, action] = 0
                int_mu[0],int_Sigma[0,0] = int_mu[0]*self.alpha1,int_Sigma[0,0]*self.alpha1**2
                int_mu[1],int_Sigma[1,1] = int_mu[1]*self.alpha2,int_Sigma[1,1]*self.alpha2**2
            else:
                self.actions_hist[self.cb, self.cc, self.ci, :] = self.total_years
        else:
            all_noise = np.repeat(all_noise[np.newaxis,:,:],np.size(self.ci),axis=0)
        int_mu = int_mu[np.newaxis,:]
        mu[:,3:6] = int_mu
        var[:,3:6, 3:6] = int_Sigma[np.newaxis,:,:]
        if mu.size == mu.shape[0]:
            mu = mu.squeeze()
            var = var.squeeze()
        if action != 4:
            mu_pred = (self.Am @ mu.transpose()).transpose()
            var_pred = self.Am @ var @ self.Am.transpose() + all_noise
        else:
            # space transformation to know max value in the transformed space
            self.max_cond = copy.copy(self.max_cond_original)
            max_cond,_ = self.ST.original_to_transformed(self.max_cond)
            mu_pred = np.array([max_cond, -0.1, -0.001, 0, 0, 0])
            var_pred = np.diag([1, 0.025**2, 0.0025**2, 0, 0, 0])
        # update with observations
        if np.any(~np.isnan(self.y[self.cc, self.current_year])):
            Ie = np.eye(6)
            Fm = self.Fm[np.newaxis,:]
            er = self.y[self.cc, self.current_year] - (Fm @ mu_pred.transpose()).squeeze() - self.inspector_std[self.inspector[self.current_year],1]
            var_xy = (Fm@var_pred@Fm.transpose()).squeeze()+ self.inspector_std[self.inspector[self.current_year],2]**2
            if mu.size > mu.squeeze().shape[0]:
                Ie = np.repeat(Ie[np.newaxis,:,:],np.size(self.ci),axis=0)
                var_xy = var_xy[:,np.newaxis,np.newaxis]
                er = er[:,np.newaxis]
            Kg = var_pred@Fm.transpose()/var_xy
            mu_pred = mu_pred + Kg.squeeze() * er
            var_pred = (Ie - Kg@Fm)@var_pred
        if (mu_pred.size == mu_pred.squeeze().shape[0]) and len(mu_pred.shape)<2:
            mu_pred = mu_pred[np.newaxis,:]
            var_pred = var_pred[np.newaxis,:,:]
        # check constraints
        if np.any(mu_pred[:,1] + 2 * np.sqrt(var_pred[:,1, 1]) > 0):
            const_ind = np.where(mu_pred[:,1] + 2 * np.sqrt(var_pred[:,1, 1]) > 0)
            if const_ind[0].shape[0] == 1:
                if mu_pred[0,1] < mu[0,1] and action > 0:
                    self.SC.d[0] = copy.copy(mu[0,1])
                mu_out, var_out = self.SC.state_constraints(mu_pred[const_ind[0][0],:], var_pred[const_ind[0][0],:,:])
                if mu_out[1] != -np.inf:
                    mu_pred[const_ind[0][0],:] = mu_out
                    var_pred[const_ind[0][0],:,:] = var_out
                else:
                    mu_pred[const_ind[0][0],1] = -1e-2
                    mu_pred[const_ind[0][0],2] = -1e-4
            else:
                for i in range(const_ind[0].shape[0]):
                    mu_out, var_out = self.SC.state_constraints(mu_pred[const_ind[0][i],:], var_pred[const_ind[0][i],:,:])
                    mu_pred[const_ind[0][i],:] = mu_out
                    var_pred[const_ind[0][i],:,:] = var_out

        # check min possible condition
        min_cond,_ = self.ST.original_to_transformed(self.min_cond)
        if np.any(mu_pred[:,0] < min_cond):
            const_ind = np.where(mu_pred[:,0] < min_cond)
            if const_ind[0].shape[0] == 1:
                mu_pred[const_ind[0][0],0] = min_cond
            else:
                for i in range(const_ind[0].shape[0]):
                    mu_pred[const_ind[0][i],0] = min_cond

        # check max possible condition
        max_cond,_ = self.ST.original_to_transformed(self.max_cond)
        if np.any(mu_pred[:,0] > max_cond):
            const_ind = np.where(mu_pred[:,0] > max_cond)
            if const_ind[0].shape[0] == 1:
                mu_pred[const_ind[0][0],0] = max_cond
            else:
                for i in range(const_ind[0].shape[0]):
                    mu_pred[const_ind[0][i],0] = max_cond

        if mu_pred.size == mu_pred.squeeze().shape[0]:
            mu_pred = mu_pred.squeeze()
            var_pred = var_pred.squeeze()
            if mu_pred[1] == -np.inf:
                mu_pred[1] =  mu[1]
        # update element estimate
        self.e_Ex[self.cb, self.cc, self.ci,:] = mu_pred
        self.e_Var[self.cb, self.cc, self.ci,:,:] = var_pred

    def kalman_smoother(self, x, Var, action):
        TotalTimeSteps = x.shape[1]
        ExSmooth = np.zeros([6,TotalTimeSteps])
        VarSmooth = np.zeros([6,6,TotalTimeSteps])
        ExSmooth[:,-1] = x[:,-1]
        VarSmooth[:,:,-1] = Var[:,:,-1]
        InterventionVector = (action!=0)
        for i in reversed(range(TotalTimeSteps - 1)):
            if InterventionVector[i+1]:
                self.Am[0:3,3:6] = np.eye(3)
                Q_int = self.int_Q[action[i]-1]
            else:
                self.Am[0:3,3:6] = np.zeros([3,3])
                Q_int = np.zeros([6,6])
            all_noise = sp.block_diag(self.Q,np.zeros([3,3])) + Q_int
            Xpred=self.Am @ x[:,i]
            Vpred= self.Am @ Var[:,:,i] @ self.Am.transpose()  + all_noise
            J = Var[:,:,i] @ self.Am.transpose() @ self.inv(Vpred)
            ExSmooth[:,i]=x[:,i] +J @ (ExSmooth[:,i+1]-Xpred)
            VarSmooth[:,:,i] = Var[:,:,i] + J @ (VarSmooth[:,:,i+1]-Vpred) @ J.transpose()
            if InterventionVector[i+1]:
                if (ExSmooth[1,i]>ExSmooth[1,i+1] or ExSmooth[1,i]+ 2 * np.sqrt(VarSmooth[1, 1, i]) > 0):
                    store_d = copy.copy(self.SC.d)
                    self.SC.d = np.array([-75.0, ExSmooth[1,i+1]])
                    ExSmooth[:, i], VarSmooth[:,:,i] = self.SC.state_constraints(ExSmooth[:, i], VarSmooth[:,:,i])
                    self.SC.d = copy.copy(store_d)

                if abs(ExSmooth[2,i])<abs(ExSmooth[2,i+1]):
                    store_D = copy.copy(self.SC.D)
                    store_d = copy.copy(self.SC.d)
                    self.SC.D = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
                    self.SC.d = np.array([-5, ExSmooth[2, i+1]+ np.sqrt(VarSmooth[2, 2, i+1]) ])
                    ExSmooth[:, i], VarSmooth[:,:,i] = self.SC.state_constraints(ExSmooth[:, i], VarSmooth[:,:,i])
                    self.SC.D = copy.copy(store_D)
                    self.SC.d = copy.copy(store_d)

            if (ExSmooth[1,i] + 2 * np.sqrt(VarSmooth[1, 1, i]) > 0) and ~InterventionVector[i+1]:
                ExSmooth[:, i], VarSmooth[:,:,i] = self.SC.state_constraints(ExSmooth[:, i], VarSmooth[:,:,i])

        return ExSmooth, VarSmooth


    def elem_action(self, action):
        reward = 0
        self.cc = self.selected_cat 
        state = self.state_element_prep()
        # perform element action
        self.estimate_step(self.e_Ex[self.cb,self.cc,self.ci,:], self.e_Var[self.cb, self.cc, self.ci, :, :], action)
        self.ec = self.ec - 1
        reward = 0
        # advance time
        self.current_year += 1
        next_state = self.state_element_prep() 
        return state, action, reward, next_state
    
    def state_element_prep(self):
        # state before action
        state_original = self.state_prep_0()
        # state before action
        state = self.assemble_state(state_original)
        return state
    
    def assemble_state(self, state):
        return state
    
    def state_prep_0(self):
        state_tr = copy.deepcopy(self.e_Ex[self.cb, self.cc, self.ci, 0:2])
        # back-transform the space
        state = self.transform_to_original(state_tr)
        return state
    
    def transform_to_original(self,state):
        or_state = np.zeros(state.shape[0])
        or_state[0] = copy.copy(self.ST.transformed_to_original(state[0]))
        or_state[1],_,_,_,_= copy.copy(self.ST.transformed_to_original_speed(state[0],state[1],np.ones(1)))
        return or_state
    
    def state_plot(self, mu, var, y, R, B, plot_type, *args):
        if plot_type == 'condition':
            mu_cond_original = self.ST.transformed_to_original(mu).copy()
            y_original_b = copy.copy(self.ST.transformed_to_original(y-B))
            y_original = copy.copy(self.ST.transformed_to_original(y))
            r_above = copy.copy(self.ST.transformed_to_original(y-B + 2 * np.sqrt(R))) 
            r_under = copy.copy(self.ST.transformed_to_original(y-B - 2 * np.sqrt(R)))
            std = np.sqrt(var)
            std_original_1p = self.ST.transformed_to_original(mu + std).copy()
            std_original_1n = self.ST.transformed_to_original(mu - std).copy()
            std_original_2p = self.ST.transformed_to_original(mu + 2 * std).copy()
            std_original_2n = self.ST.transformed_to_original(mu - 2 * std).copy()
            return mu_cond_original, y_original, y_original_b, r_above, r_under, std_original_1p, std_original_1n, std_original_2p, std_original_2n
        elif plot_type == 'speed':
            mu_cond = args[0]
            mu_dot, std_dot_1p, std_dot_1n, std_dot_2p, std_dot_2n = self.ST.transformed_to_original_speed(mu_cond,
                                                                                                            mu, var)
            return mu_dot, std_dot_1p, std_dot_1n, std_dot_2p, std_dot_2n
        
        

    def ssm_kr_predict(self, y, total_years, inspector_std, inspector, Actions):
        num_years = len(total_years)
        mu_vec = np.zeros([6,num_years])
        var_vec = np.zeros([6,6,num_years])
        obs = np.nan * np.zeros(num_years)
        R = np.nan * np.zeros(num_years)
        B = np.nan * np.zeros(num_years)
        self.reset(y, num_years, inspector_std, inspector)
        mu_vec[:, 0] = self.e_Ex[self.cb,self.cc, self.ci,0:6]
        var_vec[:,:,0] = self.e_Var[self.cb,self.cc, self.ci, 0:6,0:6]
        for k in range(1, num_years):
            state, _, _, _ = self.step(Actions[k])
            mu_vec[:, k] = self.e_Ex[self.cb,self.cc, self.ci,0:6]
            var_vec[:,:,k] = self.e_Var[self.cb,self.cc, self.ci, 0:6,0:6]
            obs[k] =  self.y[self.cc, k]
            R[k] = self.inspector_std[self.inspector[k],2]**2
            B[k] = self.inspector_std[self.inspector[k],1]
            #print(state)

        mu_vec, var_vec = self.kalman_smoother(mu_vec, var_vec, Actions)
        
        mu_cond_original, y_original, y_original_b, r_above, r_under, std_original_1p, std_original_1n, std_original_2p, std_original_2n = \
            self.state_plot(mu_vec[0,:],var_vec[0,0,:],obs,R,B,'condition')
        mu_dot, std_dot_1p, std_dot_1n, std_dot_2p, std_dot_2n = \
            self.state_plot(mu_vec[1,:],var_vec[1,1,:],obs,R,B,'speed',mu_vec[0,:])
        
        df_cond = pd.DataFrame(np.round(np.c_[mu_cond_original[None,].transpose(), total_years[None,].transpose(), 
                                y_original[None,].transpose(), y_original_b[None,].transpose(), r_above[None,].transpose(), r_under[None,].transpose(), 
                                std_original_1p[None,].transpose(), std_original_1n[None,].transpose(), 
                                std_original_2p[None,].transpose(), std_original_2n[None,].transpose()],2), 
                          columns=['Condition', 'Years', 'Inspections', 'Inspections_corrected' ,'r_above', 'r_under', 'std_1p', 'std_1n', 'std_2p', 'std_2n'])
        df_cond.Years = df_cond.Years.astype("category")
        df_speed = pd.DataFrame(np.round(np.c_[mu_dot[None,].transpose(), total_years[None,].transpose(), 
                                 std_dot_1p[None,].transpose(), std_dot_1n[None,].transpose(), 
                                 std_dot_2p[None,].transpose(), std_dot_2n[None,].transpose()],4), 
                          columns=['Speed', 'Years', 'std_1p', 'std_1n', 'std_2p', 'std_2n'])
        df_speed.Years = df_speed.Years.astype("category")
        return df_cond, df_speed
    
    def plot_results(self, df_cond, df_speed):
        line = alt.layer(
                    alt.Chart(df_cond).mark_area(opacity=0.4, color='red').encode(
                    y='std_2n',
                    y2='std_2p'
                    ),
                    alt.Chart(df_cond).mark_area(opacity=0.5, color='white').encode(
                    y='std_1n',
                    y2='std_1p'
                    ),
                    alt.Chart(df_cond).mark_line(strokeDash=[5,1],strokeWidth=1, color='white').encode(
                    y = alt.Y('Condition', scale=alt.Scale(zero=False),title='Condition')),
                    alt.Chart(df_cond).mark_point(color='red').encode(y = 'Inspections'),
                    alt.Chart(df_cond).mark_point(color='purple').encode(y = 'Inspections_corrected'),
                    alt.Chart(df_cond).mark_errorbar(color='blue', ticks=True).encode(
                    y = alt.Y('r_under', scale=alt.Scale(zero=False),title='Condition'),
                    y2 = 'r_above'),
                ).encode(
                    x = alt.X('Years',title='Time (Year)')
                ).properties(
            width=400,
            height=400
        )
        line_speed = alt.layer(
                alt.Chart(df_speed).mark_area(opacity=0.4, color='red').encode(
                    y='std_2n',
                    y2='std_2p'
                ),
                alt.Chart(df_speed).mark_area(opacity=0.5, color='white').encode(
                    y='std_1n',
                    y2='std_1p'
                ),
                alt.Chart(df_speed).mark_line(strokeDash=[5,1], color='white').encode(
                    y=alt.Y('Speed',title='Speed')),
                ).encode(
                    x = alt.X('Years',title='Time (Year)')
                ).properties(
            width=200,
            height=400
        )
        return alt.hconcat(line, line_speed)
                

class StateConstraints:
    def __init__(self):
        self.inv = np.linalg.pinv
        self.svd = np.linalg.svd
        self.D = np.array([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        self.d = np.array([-50.0, 0])
    
    def state_constraints(self, mu_kf, var_kf):
        mu_kf = mu_kf[:,np.newaxis]
        u_trunc, w_trunc, v_trunc = self.svd(var_kf)
        w_trunc = np.diag(w_trunc)
        if np.linalg.norm(u_trunc@w_trunc@u_trunc.transpose() - var_kf,2) > 1E-8:
            return mu_kf, var_kf
        amgs = np.sqrt(w_trunc) @ u_trunc.T @ np.transpose([self.D[0, :]])
        w, s = self.gram_schmidt_transformation(amgs)
        std_trunc = np.sqrt(self.D[1, :] @ var_kf @ np.transpose([self.D[1, :]]))
        s = (s * std_trunc) / w
        c_trunc = (self.d[0] - self.D[0, :] @ mu_kf) / std_trunc
        d_trunc = (self.d[1] - self.D[1, :] @ mu_kf) / std_trunc
        alpha = np.sqrt(2 / np.pi) / (mt.erf(d_trunc / np.sqrt(2)) - mt.erf(c_trunc / np.sqrt(2)) + np.finfo(float).eps)
        mu = alpha * (np.exp(-c_trunc ** 2 / 2) - np.exp(-d_trunc ** 2 / 2))
        sigma = alpha * (
                np.exp(-c_trunc ** 2 / 2) @ (c_trunc - 2 * mu) - np.exp(-d_trunc ** 2 / 2) @ (d_trunc - 2 * mu)) + mu ** 2 + 1
        mu_z = np.transpose([np.zeros(mu_kf.shape[0])])
        sigma_z = np.eye(var_kf.shape[0])
        mu_z[0, 0] = mu
        sigma_z[0, 0] = sigma
        mu_kf_new = mu_kf + u_trunc @ np.sqrt(w_trunc) @ s.T @ mu_z
        var_kf_new = u_trunc @ np.sqrt(w_trunc) @ s.T @ sigma_z @ s @ np.sqrt(w_trunc) @ u_trunc.T
        mu_kf_new = mu_kf_new.squeeze()
        return mu_kf_new, var_kf_new 

    @staticmethod
    def gram_schmidt_transformation(amgs):
        m, n = np.shape(amgs)
        w = np.zeros([n, n])
        t = np.zeros([m + n, m])
        n_range = range(n)
        for k in n_range:
            sigma = np.sqrt(amgs[:, k].T.dot(amgs[:, k]))
            if np.abs(sigma) < 100 * np.spacing(1):
                break
            w[k, k] = sigma
            for j in n_range[k + 1:]:
                w[k, j] = amgs[:, k].T.dot(amgs[:, j]) / sigma
            t[k, :] = amgs[:, k] / sigma
            for j in n_range[k + 1:]:
                amgs[:, j] = amgs[:, j] - w[k, j] * (amgs[:, k]) / sigma
        t[n:n + m, 0:m] = np.eye(m)
        index = n
        tot = range(n + m)
        for k in tot[n:]:
            temp = t[k, :]
            for i in range(k):
                temp = temp - t[k, :].dot(np.transpose([t[i, :]])).dot([t[i, :]])
            if np.linalg.norm(temp) > 100 * np.spacing(1):
                t[index, :] = temp / np.linalg.norm(temp)
                index = index + 1
        T = t[0:m, 0:m]
        return w, T
    
class SpaceTransformation:
    def __init__(self, n, upper_limit, lower_limit):
        self.n = 2 ** n
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.max_x = sc.gammaincinv(1 / self.n, 0.999) ** (1 / self.n)

    def original_to_transformed(self, y_original):
        y_tr = np.zeros(y_original.size)
        y_tr_s = np.zeros(y_original.size)
        if y_original.size == 1:
            y_tr, y_tr_s = self.compute_y_tr(y_original)
        else:
            for i in range(y_original.size):
                y_tr[i], y_tr_s[i] = self.compute_y_tr(y_original[i])
        return y_tr, y_tr_s

    def transformed_to_original(self, y_trans):
        y = np.zeros(y_trans.size)
        if y.shape[0] == 1:
            y = self.compute_y(y_trans)
        else:
            for i in range(y_trans.shape[0]):
                y[i] = self.compute_y(y_trans[i])
        return y
            

    def transformed_to_original_speed(self, mu, mu_dot, var_dot):
        max_range = (self.upper_limit - self.lower_limit) - (self.upper_limit - self.lower_limit) / self.max_x
        fy_tr = interpolate.interp1d([self.lower_limit - max_range, self.upper_limit + max_range], [-self.max_x, self.max_x])
        std_dot = np.sqrt(var_dot)
        mu_dot_original = np.zeros(mu_dot.size)
        std_dot_1p = np.zeros(mu_dot.size)
        std_dot_1n = np.zeros(mu_dot.size)
        std_dot_2p = np.zeros(mu_dot.size)
        std_dot_2n = np.zeros(mu_dot.size)
        if np.size(mu) == 1:
            if mu > self.upper_limit + max_range:
                mu = self.upper_limit + max_range
            if mu < self.lower_limit - max_range:
                mu = self.lower_limit - max_range
        else:
            if any(mu > self.upper_limit + max_range):
                ind_ = np.where(mu > self.upper_limit + max_range)
                mu[ind_] = self.upper_limit + max_range
            if any(mu < self.lower_limit - max_range):
                ind_ = np.where(mu < self.lower_limit - max_range)
                mu[ind_] = self.lower_limit - max_range
        if mu_dot.size==1:
            mu_s = fy_tr(mu)
            mu_dot_original = mu_dot * self.derivative_g(mu_s, self.n)
            std_dot_1p = (mu_dot + std_dot) * self.derivative_g(mu_s, self.n)
            std_dot_1n = (mu_dot - std_dot) * self.derivative_g(mu_s, self.n)
            std_dot_2p = (mu_dot + 2 * std_dot) * self.derivative_g(mu_s, self.n)
            std_dot_2n = (mu_dot - 2 * std_dot) * self.derivative_g(mu_s, self.n)
        else:
            for i in range(mu_dot.size):
                if mu[i] < self.lower_limit - max_range:
                    mu[i] = self.lower_limit - max_range
                mu_s = fy_tr(mu[i])
                mu_dot_original[i] = mu_dot[i] * self.derivative_g(mu_s, self.n)
                std_dot_1p[i] = (mu_dot[i] + std_dot[i]) * self.derivative_g(mu_s, self.n)
                std_dot_1n[i] = (mu_dot[i] - std_dot[i]) * self.derivative_g(mu_s, self.n)
                std_dot_2p[i] = (mu_dot[i] + 2 * std_dot[i]) * self.derivative_g(mu_s, self.n)
                std_dot_2n[i] = (mu_dot[i] - 2 * std_dot[i]) * self.derivative_g(mu_s, self.n)
        return mu_dot_original, std_dot_1p, std_dot_1n, std_dot_2p, std_dot_2n

    @staticmethod
    def derivative_g(x, n):
        return n * np.exp(-x ** n) / sc.gamma(1 / n)

    def compute_y_tr(self, y_original):
        max_range = (self.upper_limit - self.lower_limit) - (self.upper_limit - self.lower_limit) / self.max_x
        fy_tr = interpolate.interp1d([-self.max_x, self.max_x], [self.lower_limit - max_range, self.upper_limit + max_range])
        fy_s = interpolate.interp1d([self.lower_limit, self.upper_limit], [-1, 1])
        if y_original > self.upper_limit:
            y_s = 0.999
            y_tr_s = sc.gammaincinv(
                1 / self.n, np.abs(y_s)) ** (1 / self.n)
            y_tr = fy_tr(y_tr_s)
        elif y_original > (self.upper_limit - self.lower_limit) / 2 + self.lower_limit:
            if y_original == self.upper_limit:
                y_original = self.upper_limit.copy() - 0.0001
            y_s = fy_s(y_original)
            if y_s > 0.999:
                y_s = 0.999
            y_tr_s = sc.gammaincinv(1 / self.n, np.abs(y_s)) ** (1 / self.n)
            if y_tr_s > self.max_x:
                y_tr_s = self.max_x.copy()
            y_tr = fy_tr(y_tr_s)
        elif y_original == (self.upper_limit - self.lower_limit) / 2 + self.lower_limit:
            y_tr_s = 0
            y_tr = fy_tr(y_tr_s)
        elif y_original < (self.upper_limit - self.lower_limit) / 2 + self.lower_limit:
            if y_original == self.lower_limit:
                y_original = self.lower_limit + 0.0001
            y_s = fy_s(y_original)
            y_tr_s = -sc.gammaincinv(1 / self.n, np.abs(y_s)) ** (1 / self.n)
            if y_tr_s < -self.max_x:
                y_tr_s = -self.max_x.copy()
            y_tr = fy_tr(y_tr_s)
        else:
            y_tr_s = np.NaN
            y_tr = np.NaN
        return y_tr, y_tr_s

    def compute_y(self, y_trans):
        max_range = (self.upper_limit - self.lower_limit) - (self.upper_limit - self.lower_limit) / self.max_x
        fy_tr = interpolate.interp1d([self.lower_limit - max_range, self.upper_limit + max_range], [-self.max_x, self.max_x])
        fy_s = interpolate.interp1d([-1, 1], [self.lower_limit, self.upper_limit])
        y_s = np.zeros(y_trans.size)
        if y_trans > self.upper_limit + max_range:
            y_trans = self.upper_limit + max_range
        elif y_trans < self.lower_limit - max_range:
            y_trans = self.lower_limit - max_range
        y_s = fy_tr(y_trans)
        if y_s > self.max_x:
            y_s = self.max_x.copy()
            y_s_tr = sc.gammainc(1 / self.n, y_s ** self.n)
            y = fy_s(y_s_tr)
        elif y_s > 0:
            y_s_tr = sc.gammainc(1 / self.n, y_s ** self.n)
            y = fy_s(y_s_tr)
        elif y_s < -self.max_x:
            y_s = -self.max_x.copy()
            y_s_tr = -sc.gammainc(1 / self.n, y_s ** self.n)
            y = fy_s(y_s_tr)
        elif y_s == 0:
            y = (self.upper_limit - self.lower_limit)/2 + self.lower_limit
        elif y_s < 0:
            y_s_tr = -sc.gammainc(1 / self.n, y_s ** self.n)
            y = fy_s(y_s_tr)
        else:
            y = np.NaN
        return y
    
class RegressionModel:
    def __init__(self) -> None:
        pass
    def CalculateDistance(self, XN, XM):
        Positive = 0
        Distance = np.sum(np.power(XN,2)) -2*((XN*np.transpose([XM])).squeeze()) + np.power(XM,2)
        if Positive: 
            Distance = np.max(0, Distance)
        return Distance

    def Kernel_Function(self, x, y, Param, KernelType):
        N = np.size(x)
        M = np.shape(y)[0]
        if KernelType == 'RBF':
            K = self.CalculateDistance(np.divide(x,Param),np.divide(y,Param))
            K=np.exp(-0.5*(K))
        elif KernelType ==  'AAK':
            # Aitchison and Aitken (1976) unordered discrete kernel
            h = Param
            num_levels = len(np.unique(y))
            K = np.multiply(np.ones((N,M)), h)/(num_levels - 1)
            Ind = np.where(y==x)[0]
            K[0, Ind] = (1 - h)

        elif KernelType == 'Matern12': # Matern32 kernel
            K = np.zeros((N,M))
            K = K + self.CalculateDistance(x/Param,y/Param)
            K = np.sqrt(K)
            K = np.exp(-K)
        elif KernelType == 'Matern32': # Matern32 kernel
            K = np.zeros((N,M))
            for i in range(len(Param)):
                K = K + self.CalculateDistance(x[i]/Param(i),y[i]/Param[i])
            K=np.sqrt(3)*np.sqrt(K)
            K=np.multiply((1 + K),np.exp(-K))
        elif KernelType == 'Matern52': # Matern52 kernel
            K = np.zeros((N,M))
            K = self.CalculateDistance(x/Param, y/Param)
            K = np.sqrt(5)*np.sqrt(K)
            K = np.multiply(1 + np.multiply(K, (1 + K/3)),np.exp(-K))
        elif KernelType == 'Laplace': # Laplace kernel
            K = np.zeros((N,M))
            for i in range(len(Param)):
                K = K + np.sqrt(self.CalculateDistance(x[:,i],y[:,i]))/(2 * Param ** 2)
            K=np.exp(-K)
        elif KernelType == 'Polynomial': # Polynomial kernel
            a = Param[1]   #  order
            b = Param[2]   #  constant
            K = np.power(x@np.transpose([y]) + b,a)
        elif KernelType == 'Linear': # Linear kernel
            K = x@np.transpose([y])
        return K
    
class SpaceTransformation:
    def __init__(self, n, upper_limit, lower_limit):
        self.n = 2 ** n
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.max_x = sc.gammaincinv(1 / self.n, 0.999) ** (1 / self.n)

    def original_to_transformed(self, y_original):
        y_tr = np.zeros(y_original.size)
        y_tr_s = np.zeros(y_original.size)
        if y_original.size == 1:
            y_tr, y_tr_s = self.compute_y_tr(y_original)
        else:
            for i in range(y_original.size):
                y_tr[i], y_tr_s[i] = self.compute_y_tr(y_original[i])
        return y_tr, y_tr_s

    def transformed_to_original(self, y_trans):
        y = np.zeros(y_trans.size)
        if y.shape[0] == 1:
            y = self.compute_y(y_trans)
        else:
            for i in range(y_trans.shape[0]):
                y[i] = self.compute_y(y_trans[i])
        return y
            

    def transformed_to_original_speed(self, mu, mu_dot, var_dot):
        max_range = (self.upper_limit - self.lower_limit) - (self.upper_limit - self.lower_limit) / self.max_x
        fy_tr = interpolate.interp1d([self.lower_limit - max_range, self.upper_limit + max_range], [-self.max_x, self.max_x])
        std_dot = np.sqrt(var_dot)
        mu_dot_original = np.zeros(mu_dot.size)
        std_dot_1p = np.zeros(mu_dot.size)
        std_dot_1n = np.zeros(mu_dot.size)
        std_dot_2p = np.zeros(mu_dot.size)
        std_dot_2n = np.zeros(mu_dot.size)
        if np.size(mu) == 1:
            if mu > self.upper_limit + max_range:
                mu = self.upper_limit + max_range
            if mu < self.lower_limit - max_range:
                mu = self.lower_limit - max_range
        else:
            if any(mu > self.upper_limit + max_range):
                ind_ = np.where(mu > self.upper_limit + max_range)
                mu[ind_] = self.upper_limit + max_range
            if any(mu < self.lower_limit - max_range):
                ind_ = np.where(mu < self.lower_limit - max_range)
                mu[ind_] = self.lower_limit - max_range
        if mu_dot.size==1:
            mu_s = fy_tr(mu)
            mu_dot_original = mu_dot * self.derivative_g(mu_s, self.n)
            std_dot_1p = (mu_dot + std_dot) * self.derivative_g(mu_s, self.n)
            std_dot_1n = (mu_dot - std_dot) * self.derivative_g(mu_s, self.n)
            std_dot_2p = (mu_dot + 2 * std_dot) * self.derivative_g(mu_s, self.n)
            std_dot_2n = (mu_dot - 2 * std_dot) * self.derivative_g(mu_s, self.n)
        else:
            for i in range(mu_dot.size):
                if mu[i] < self.lower_limit - max_range:
                    mu[i] = self.lower_limit - max_range
                if mu[i] > self.upper_limit + max_range:
                    mu[i] = self.upper_limit + max_range
                mu_s = fy_tr(mu[i])
                mu_dot_original[i] = mu_dot[i] * self.derivative_g(mu_s, self.n)
                std_dot_1p[i] = (mu_dot[i] + std_dot[i]) * self.derivative_g(mu_s, self.n)
                std_dot_1n[i] = (mu_dot[i] - std_dot[i]) * self.derivative_g(mu_s, self.n)
                std_dot_2p[i] = (mu_dot[i] + 2 * std_dot[i]) * self.derivative_g(mu_s, self.n)
                std_dot_2n[i] = (mu_dot[i] - 2 * std_dot[i]) * self.derivative_g(mu_s, self.n)
        return mu_dot_original, std_dot_1p, std_dot_1n, std_dot_2p, std_dot_2n

    @staticmethod
    def derivative_g(x, n):
        return n * np.exp(-x ** n) / sc.gamma(1 / n)

    def compute_y_tr(self, y_original):
        max_range = (self.upper_limit - self.lower_limit) - (self.upper_limit - self.lower_limit) / self.max_x
        fy_tr = interpolate.interp1d([-self.max_x, self.max_x], [self.lower_limit - max_range, self.upper_limit + max_range])
        fy_s = interpolate.interp1d([self.lower_limit, self.upper_limit], [-1, 1])
        if y_original > self.upper_limit:
            y_s = 0.999
            y_tr_s = sc.gammaincinv(
                1 / self.n, np.abs(y_s)) ** (1 / self.n)
            y_tr = fy_tr(y_tr_s)
        elif y_original > (self.upper_limit - self.lower_limit) / 2 + self.lower_limit:
            if y_original == self.upper_limit:
                y_original = self.upper_limit.copy() - 0.0001
            y_s = fy_s(y_original)
            if y_s > 0.999:
                y_s = 0.999
            y_tr_s = sc.gammaincinv(1 / self.n, np.abs(y_s)) ** (1 / self.n)
            if y_tr_s > self.max_x:
                y_tr_s = self.max_x.copy()
            y_tr = fy_tr(y_tr_s)
        elif y_original == (self.upper_limit - self.lower_limit) / 2 + self.lower_limit:
            y_tr_s = 0
            y_tr = fy_tr(y_tr_s)
        elif y_original < (self.upper_limit - self.lower_limit) / 2 + self.lower_limit:
            if y_original == self.lower_limit:
                y_original = self.lower_limit + 0.0001
            y_s = fy_s(y_original)
            y_tr_s = -sc.gammaincinv(1 / self.n, np.abs(y_s)) ** (1 / self.n)
            if y_tr_s < -self.max_x:
                y_tr_s = -self.max_x.copy()
            y_tr = fy_tr(y_tr_s)
        else:
            y_tr_s = np.NaN
            y_tr = np.NaN
        return y_tr, y_tr_s

    def compute_y(self, y_trans):
        max_range = (self.upper_limit - self.lower_limit) - (self.upper_limit - self.lower_limit) / self.max_x
        fy_tr = interpolate.interp1d([self.lower_limit - max_range, self.upper_limit + max_range], [-self.max_x, self.max_x])
        fy_s = interpolate.interp1d([-1, 1], [self.lower_limit, self.upper_limit])
        y_s = np.zeros(y_trans.size)
        if y_trans > self.upper_limit + max_range:
            y_trans = self.upper_limit + max_range
        elif y_trans < self.lower_limit - max_range:
            y_trans = self.lower_limit - max_range
        y_s = fy_tr(y_trans)
        if y_s > self.max_x:
            y_s = self.max_x.copy()
            y_s_tr = sc.gammainc(1 / self.n, y_s ** self.n)
            y = fy_s(y_s_tr)
        elif y_s > 0:
            y_s_tr = sc.gammainc(1 / self.n, y_s ** self.n)
            y = fy_s(y_s_tr)
        elif y_s < -self.max_x:
            y_s = -self.max_x.copy()
            y_s_tr = -sc.gammainc(1 / self.n, y_s ** self.n)
            y = fy_s(y_s_tr)
        elif y_s == 0:
            y = (self.upper_limit - self.lower_limit)/2 + self.lower_limit
        elif y_s < 0:
            y_s_tr = -sc.gammainc(1 / self.n, y_s ** self.n)
            y = fy_s(y_s_tr)
        else:
            y = np.NaN
        return y
        