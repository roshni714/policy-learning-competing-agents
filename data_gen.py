import pandas as pd
import numpy as np
from utils_nels import impute_values
from scipy.stats import norm
from agent_distribution import AgentDistribution
import random
from utils import compute_continuity_noise_gammas
from sklearn.cluster import KMeans


def generate_covariates():
    stmeg_variables = [
        "STU_ID",
        "F2SES1",  # SOCIO-ECONOMIC STATUS COMPOSITE
        "F22XRSTD",  # READING STANDARDIZED SCORE
        "F22XMSTD",  # MATHEMATICS STANDARDIZED SCORE
        "F22XSSTD",  # SCIENCE STANDARDIZED SCORE
        "F22XHSTD",  # HISTORY/CIT/GEOG STANDARDIZED SCORE
        "F2RHENG2",  # AVERAGE GRADE IN ENGLISH (HS+B)
        "F2RHMAG2",  # AVERAGE GRADE IN MATHEMATICS (HS+B)
        "F2RHSCG2",  # AVERAGE GRADE IN SCIENCE (HS+B)
        "F2RHSOG2",  # AVERAGE GRADE IN SOCIAL STUDIES (HS+B)
        # "F2RHCOG2", #AVERAGE GRADE IN COMP. SCIENCE (HS+B)
        "F2RHFOG2",  # AVERAGE GRADE IN FOREIGN LANG. (HS+B)
    ]

    stmeg = pd.read_csv("data/nels_88_94_stmeg3_v1_0.csv", usecols=stmeg_variables)
    stmeg = stmeg.replace(r"^\s*$", np.nan, regex=True)
    stmeg = stmeg.astype(
        {  #'F2RGPA': 'float32',
            "F2RHENG2": "float32",
            "F2RHMAG2": "float32",
            "F2RHSCG2": "float32",
            "F2RHSOG2": "float32",
            # "F2RHCOG2": 'float32',
            "F2RHFOG2": "float32",
        }
    )

    to_replace = {
        "F2SES1": [99.998],
        "F22XRSTD": [99.998, 99.999],
        "F22XMSTD": [99.998, 99.999],
        "F22XSSTD": [99.998, 99.999],
        "F22XHSTD": [99.998, 99.999],
        "F2RHENG2": [99.98, np.nan],
        "F2RHMAG2": [99.98, np.nan],
        "F2RHSCG2": [99.98, np.nan],
        "F2RHSOG2": [99.98, np.nan],
        # "F2RHCOG2": [99.98, np.nan],
        "F2RHFOG2": [99.98, np.nan],
    }

    replacement_vals = {
        "F2SES1": -0.088,
        "F22XRSTD": 63.81,
        "F22XMSTD": 63.96,
        "F22XSSTD": 64.01,
        "F22XHSTD": 64.30,
        "F2RHENG2": 7.07,
        "F2RHMAG2": 7.61,
        "F2RHSCG2": 7.43,
        "F2RHSOG2": 7.01,
        # "F2RHCOG2" : 5.78,
        "F2RHFOG2": 6.58,
    }

    min_val = {
        "F2SES1": -3.243,
        "F22XRSTD": 0.0,
        "F22XMSTD": 0.0,
        "F22XSSTD": 0.0,
        "F22XHSTD": 0.0,
        "F2RHENG2": 1.0,
        "F2RHMAG2": 1.0,
        "F2RHSCG2": 1.0,
        "F2RHSOG2": 1.0,
        # "F2RHCOG2" : 1.,
        "F2RHFOG2": 1.0,
    }

    max_val = {
        "F2SES1": 2.743,
        "F22XRSTD": 100.0,
        "F22XMSTD": 100.0,
        "F22XSSTD": 100.0,
        "F22XHSTD": 100.0,
        "F2RHENG2": 13.0,
        "F2RHMAG2": 13.0,
        "F2RHSCG2": 13.0,
        "F2RHSOG2": 13.0,
        # "F2RHCOG2" : 13.,
        "F2RHFOG2": 13,
    }

    impute_values(stmeg, to_replace, replacement_vals)

    for variable in stmeg_variables:
        if variable not in ["STU_ID", "F2SES1"]:
            stmeg[variable] = (
                10
                * (stmeg[variable] - min_val[variable])
                / (max_val[variable] - min_val[variable])
            )
        if variable == "F2SES1":
            stmeg[variable] = (stmeg[variable] - min_val[variable]) / (
                max_val[variable] - min_val[variable]
            )
        if variable.startswith("F2RH"):
            stmeg[variable] = 10 - stmeg[variable]
    X = stmeg[stmeg_variables[2:]].to_numpy()
    socio_econ = stmeg[stmeg_variables[1:2]].to_numpy()
    stu_id = stmeg[stmeg_variables[:1]].to_numpy()
    return X, socio_econ, stu_id


def compute_gammas(socio_econ):
    return (1 - np.clip(socio_econ, 0.0, 1.0) + 0.05) * 0.1


def compute_etas(X, gammas, sigma, beta, s):

    probs = norm.pdf(s - np.dot(X, beta), 0, scale=sigma).reshape(X.shape[0], 1)

    etas_1 = X - (1 / (2 * gammas)) * probs * beta
    etas_2 = X - (1 / (2 * 5)) * probs * beta
    etas = np.copy(etas_2)
    etas[:, 4:] = np.copy(etas_1[:, 4:])
    return etas


def generate_month_attended_losses():
    stmeg_variables = ["STU_ID", "F3ATTEND"]
    stmeg = pd.read_csv("data/nels_88_94_stmeg3_v1_0.csv", usecols=stmeg_variables)
    stmeg = stmeg.replace(r"^\s*$", np.nan, regex=True)
    stmeg = stmeg.astype({"F3ATTEND": "float32"})

    stmeg["F3ATTEND"].replace(to_replace=-9.0, value=0.0, inplace=True)
    stmeg["F3ATTEND"].replace(to_replace=-6.0, value=np.nan, inplace=True)
    stmeg["F3ATTEND"].replace(
        to_replace=np.nan, value=stmeg["F3ATTEND"].mean(), inplace=True
    )

    loss_admitted = -stmeg["F3ATTEND"].to_numpy().flatten()
    #     _, socio_econ, _ = generate_covariates()
    #     loss_admitted = -stmeg["F3ATTEND"].to_numpy()

    stu_id = stmeg["STU_ID"].to_numpy()

    return loss_admitted.reshape(len(loss_admitted), 1, 1), stu_id


def generate_hrs_work_losses():
    stmeg_variables = ["STU_ID", "HRSWORK1"]
    stmeg = pd.read_csv("data/nels_88_94_stmeg3_v1_0.csv", usecols=stmeg_variables)
    stmeg = stmeg.replace(r"^\s*$", np.nan, regex=True)
    stmeg = stmeg.astype({"HRSWORK1": "float32"})

    stmeg["HRSWORK1"].replace(to_replace=-9.0, value=0.0, inplace=True)
    for x in [-8.0, -7.0, -6.0, -3.0]:
        stmeg["HRSWORK1"].replace(to_replace=x, value=np.nan, inplace=True)
    stmeg["HRSWORK1"].replace(
        to_replace=np.nan, value=stmeg["HRSWORK1"].mean(), inplace=True
    )

    hrs_work = -stmeg["HRSWORK1"].to_numpy().flatten()
    stu_id = stmeg["STU_ID"].to_numpy()

    return hrs_work.reshape(len(hrs_work), 1, 1), stu_id


def get_types_and_noise(prev_beta, seed=0):

    np.random.seed(seed)

    X, socio_econ, stu_id = generate_covariates()

    scores = [np.dot(prev_beta, X[i]) for i in range(len(X))]
    prev_s = np.quantile(scores, 0.25)
    gammas = compute_gammas(socio_econ)
    sigma = compute_continuity_noise_gammas(gammas)
    etas = compute_etas(X, gammas, sigma, prev_beta, prev_s)
    etas = etas.reshape(etas.shape[0], etas.shape[1], 1)
    gammas = gammas.reshape(etas.shape[0], 1, 1)  # * np.ones(etas.shape)
    all_types = np.concatenate((etas, gammas), axis=1)
    all_types = all_types.reshape(all_types.shape[0], all_types.shape[1])

    return all_types, etas, gammas, sigma


def get_types_loss_and_noise(prev_beta, seed=0):

    np.random.seed(seed)

    X, socio_econ, stu_id = generate_covariates()

    scores = [np.dot(prev_beta, X[i]) for i in range(len(X))]
    prev_s = np.quantile(scores, 0.7)
    gammas = compute_gammas(socio_econ)
    sigma = compute_continuity_noise_gammas(gammas)
    etas = compute_etas(X, gammas, sigma, prev_beta, prev_s)
    etas = etas.reshape(etas.shape[0], etas.shape[1], 1)
    gammas = gammas.reshape(etas.shape[0], 1, 1)  # * np.ones(etas.shape)
    losses, _ = generate_month_attended_losses()
    beta_only_test = np.ones(prev_beta.shape)
    beta_only_test[4:] = 0.0
    hrs_work_losses, _ = generate_hrs_work_losses()

    # eta_losses = -np.dot(etas.reshape(etas.shape[0], etas.shape[1]), beta_only_test).reshape(losses.shape)
    all_types_and_losses = np.concatenate(
        (etas, gammas, losses, hrs_work_losses), axis=1
    )
    all_types_and_losses = all_types_and_losses.reshape(
        all_types_and_losses.shape[0], all_types_and_losses.shape[1]
    )

    return all_types_and_losses, losses, hrs_work_losses, sigma


def get_agent_distribution_and_losses_nels(n, prev_beta, n_clusters=10, seed=0):

    np.random.seed(seed)

    all_types_and_losses, _, _, sigma = get_types_loss_and_noise(prev_beta, seed)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_types_and_losses)
    all_labels = kmeans.predict(all_types_and_losses)
    unique, counts = np.unique(all_labels, return_counts=True)
    prop = counts / all_types_and_losses.shape[0]
    center_clusters = kmeans.cluster_centers_
    rep_etas = kmeans.cluster_centers_[:, :-3].reshape(
        n_clusters, all_types_and_losses.shape[1] - 3, 1
    )
    rep_gammas = kmeans.cluster_centers_[:, -3:-2].reshape(n_clusters, 1, 1) * np.ones(
        (n_clusters, all_types_and_losses.shape[1] - 3, 1)
    )
    rep_gammas[:, 4:, :] = 1.0

    month_attended_losses = kmeans.cluster_centers_[:, -2:-1].reshape(n_clusters, 1, 1)
    hrs_work_losses = kmeans.cluster_centers_[:, -1].reshape(n_clusters, 1, 1)
    #    import pdb
    #    pdb.set_trace()
    agent_dist = AgentDistribution(
        n=n,
        d=9,
        n_types=n_clusters,
        types={"etas": rep_etas, "gammas": rep_gammas},
        prop=prop,
    )
    return (
        agent_dist,
        all_types_and_losses,
        all_labels,
        month_attended_losses,
        hrs_work_losses,
        sigma,
    )


def get_agent_distribution_nels(n, prev_beta, n_clusters=10, seed=0):

    np.random.seed(seed)

    all_types, sigma = get_types_and_noise(prev_beta, seed)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_types)
    all_labels = kmeans.predict(all_types)
    unique, counts = np.unique(all_labels, return_counts=True)
    prop = counts / all_types.shape[0]
    center_clusters = kmeans.cluster_centers_
    rep_etas = kmeans.cluster_centers_[:, :-1].reshape(
        n_clusters, all_types.shape[1] - 1, 1
    )
    rep_gammas = np.ones((n_clusters, all_types.shape[1] - 1, 1))

    agent_dist = AgentDistribution(
        n=n,
        d=9,
        n_types=n_clusters,
        types={"etas": rep_etas, "gammas": rep_gammas},
        prop=prop,
    )
    return agent_dist, all_types, all_labels, sigma
