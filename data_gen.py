import pandas as pd
import numpy as np
from utils_nels import impute_values
from scipy.stats import norm
from agent_distribution import AgentDistribution
import random
from utils import compute_continuity_noise_gammas
from sklearn.cluster import KMeans

def generate_covariates():
    stmeg_variables = ["STU_ID",
                       "F2SES1", #SOCIO-ECONOMIC STATUS COMPOSITE
                       "F22XRSTD", #READING STANDARDIZED SCORE
                       "F22XMSTD", #MATHEMATICS STANDARDIZED SCORE
                       "F22XSSTD", #SCIENCE STANDARDIZED SCORE
                       "F22XHSTD", #HISTORY/CIT/GEOG STANDARDIZED SCORE
                       "F2RHENG2", #AVERAGE GRADE IN ENGLISH (HS+B)
                       "F2RHMAG2", #AVERAGE GRADE IN MATHEMATICS (HS+B)
                       "F2RHSCG2", #AVERAGE GRADE IN SCIENCE (HS+B)
                       "F2RHSOG2", #AVERAGE GRADE IN SOCIAL STUDIES (HS+B)
                       #"F2RHCOG2", #AVERAGE GRADE IN COMP. SCIENCE (HS+B)
                       "F2RHFOG2", #AVERAGE GRADE IN FOREIGN LANG. (HS+B)
                  ]
    
    stmeg = pd.read_csv("data/nels_88_94_stmeg3_v1_0.csv", usecols=stmeg_variables)
    stmeg = stmeg.replace(r'^\s*$', np.nan, regex=True)
    stmeg = stmeg.astype({ #'F2RGPA': 'float32', 
                      "F2RHENG2": 'float32',
                      "F2RHMAG2": 'float32',
                      "F2RHSCG2": 'float32',
                      "F2RHSOG2": 'float32',
                      #"F2RHCOG2": 'float32',
                      "F2RHFOG2": 'float32'})
    
    to_replace = {"F2SES1": [99.998],
                  "F22XRSTD" : [99.998, 99.999],
                  "F22XMSTD" : [99.998, 99.999],
                  "F22XSSTD": [99.998, 99.999],
                  "F22XHSTD" : [99.998, 99.999],
                  "F2RHENG2": [99.98, np.nan],
                  "F2RHMAG2": [99.98, np.nan],
                  "F2RHSCG2": [99.98, np.nan],
                  "F2RHSOG2": [99.98, np.nan],
                  #"F2RHCOG2": [99.98, np.nan],
                  "F2RHFOG2": [99.98, np.nan]}
    
    replacement_vals = {"F2SES1": - 0.088,
                        "F22XRSTD" : 63.81, 
                        "F22XMSTD": 63.96,
                        "F22XSSTD" : 64.01,
                        "F22XHSTD" : 64.30, 
                        "F2RHENG2" : 7.07,
                        "F2RHMAG2" : 7.61, 
                        "F2RHSCG2" : 7.43,
                        "F2RHSOG2" : 7.01,
                        #"F2RHCOG2" : 5.78, 
                        "F2RHFOG2" : 6.58}
    
    min_val = {"F2SES1": -3.243,
               "F22XRSTD" :0., 
               "F22XMSTD": 0.,
               "F22XSSTD" : 0.,
               "F22XHSTD" : 0., 
               "F2RHENG2" : 1.,
               "F2RHMAG2" : 1., 
               "F2RHSCG2" : 1.,
               "F2RHSOG2" : 1.,
               #"F2RHCOG2" : 1., 
               "F2RHFOG2" :1.
    }
    
    max_val = {"F2SES1": 2.743,
               "F22XRSTD" : 100., 
               "F22XMSTD": 100.,
               "F22XSSTD" : 100.,
               "F22XHSTD" : 100., 
               "F2RHENG2" : 13.,
               "F2RHMAG2" : 13., 
               "F2RHSCG2" : 13.,
               "F2RHSOG2" : 13.,
               #"F2RHCOG2" : 13., 
               "F2RHFOG2" : 13,
    }
    
    impute_values(stmeg, to_replace, replacement_vals)
    
    for variable in stmeg_variables:
        if variable != "STU_ID":
            stmeg[variable] = (stmeg[variable] - min_val[variable])/(max_val[variable] - min_val[variable])
        if variable.startswith("F2RH"):
            stmeg[variable] = -stmeg[variable]
    X = stmeg[stmeg_variables[2:]].to_numpy()
    socio_econ = stmeg[stmeg_variables[1:2]].to_numpy()
    stu_id = stmeg[stmeg_variables[:1]].to_numpy()
    return X, socio_econ, stu_id

def compute_gammas(socio_econ):
    return 1/(socio_econ + 1e-1)

def compute_etas(X, gammas, sigma, beta, s):
    etas = X + (1/(2 * gammas)) * norm.pdf(s - np.dot(X, beta), 0, scale=sigma).reshape(X.shape[0], 1) * beta
    return etas


def generate_losses():
    stmeg_variables = ["STU_ID", "F3ATTEND"]
    stmeg = pd.read_csv("data/nels_88_94_stmeg3_v1_0.csv", usecols=stmeg_variables)
    stmeg = stmeg.replace(r'^\s*$', np.nan, regex=True)
    stmeg = stmeg.astype({"F3ATTEND": 'float32'})
    
    stmeg["F3ATTEND"].replace(to_replace= -9., value=0., inplace=True)
    stmeg["F3ATTEND"].replace(to_replace= -6., value=np.nan, inplace=True)
    stmeg["F3ATTEND"].replace(to_replace=np.nan, value=stmeg["F3ATTEND"].mean(), inplace=True)
    
    stmeg["F3ATTEND"] = -stmeg["F3ATTEND"]/27
    loss_admitted = stmeg["F3ATTEND"].to_numpy()
    stu_id = stmeg["STU_ID"].to_numpy()
    
    return loss_admitted.reshape(len(loss_admitted), 1), stu_id

def get_types_and_noise(prev_beta, prev_s, seed=0):
    
    np.random.seed(seed)
    
    X, socio_econ, stu_id = generate_covariates()
    gammas = compute_gammas(socio_econ)
    sigma = compute_continuity_noise_gammas(gammas)
    print(sigma)
    etas = compute_etas(X, gammas, sigma, prev_beta, prev_s)
    etas = etas.reshape(etas.shape[0], etas.shape[1], 1)
    gammas = gammas.reshape(etas.shape[0], 1, 1) #* np.ones(etas.shape)
    print(gammas.shape)
    all_types = np.concatenate((etas, gammas), axis=1)
    all_types= all_types.reshape(all_types.shape[0], all_types.shape[1])
    
    return all_types, sigma


def get_agent_distribution_nels(n, prev_beta, prev_s, n_clusters=5, seed=0):
    
    np.random.seed(seed)
    
    all_types, sigma = get_types_and_noise(prev_beta, prev_s, seed)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_types)
    all_labels = kmeans.predict(all_types)
    unique, counts = np.unique(all_labels, return_counts=True)
    prop = counts/all_types.shape[0]
    center_clusters = kmeans.cluster_centers_
    rep_etas = kmeans.cluster_centers_[:, :-1].reshape(n_clusters, all_types.shape[1]-1, 1)
    rep_gammas = kmeans.cluster_centers_[:, -1:].reshape(n_clusters, 1, 1) * np.ones((n_clusters, all_types.shape[1]-1, 1))
    
    agent_dist = AgentDistribution(n=n, d=9, n_types=n_clusters, types={"etas":rep_etas, "gammas":rep_gammas}, prop=prop)
    return agent_dist, all_types, all_labels, sigma