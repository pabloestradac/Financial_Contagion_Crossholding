import numpy as np
import pandas as pd
import copy


def cascade_failures(dependency_matrix, theta_value, assets_price,
                     list_firms, pct, costpct):
    """
    Calculating:
    1. Total firms affected in the cascade of defaults
    2. Total costs of defaults
    3. ID of firms affected
    """
    n_total = dependency_matrix.shape[0]
    threshold_value = theta_value * np.dot(dependency_matrix, assets_price)
    current_price = copy.deepcopy(assets_price)
    beta_value = threshold_value * costpct

    list_failed_firms = []
    wave = 0
    all_failure_indicator = np.zeros((n_total, 1))
    total_costs = 0

    while (wave == 0) or (new_failed_firms.size != 0):
        wave += 1

        if wave == 1:
            all_failure_indicator[list_firms] = 1
            current_price[list_firms] = current_price[list_firms] * pct
            new_failed_firms = np.where(all_failure_indicator == 1)[0]

        else:
            fail_one = (np.dot(dependency_matrix, current_price)
                        < threshold_value).astype(int)
            fail_two = (all_failure_indicator == 0).astype(int)
            new_failure_indicator = fail_one * fail_two

            all_failure_indicator = np.maximum(all_failure_indicator,
                                               new_failure_indicator)
            total_costs = total_costs + (new_failure_indicator
                                         * beta_value).sum()
            current_price = current_price - (new_failure_indicator*beta_value)

            new_failed_firms = np.where(new_failure_indicator == 1)[0]

        list_failed_firms.append(new_failed_firms)
    total_firms = all_failure_indicator.sum() - len(list_firms)

    return total_firms, total_costs, list_failed_firms


def result_failures(theta, drop_pct, failcost_pct, econ_sector,
                    G, A, pnorm, names, n, group=False):
    """
    Return a dataframe with firm(s) that received the shock,
    and the total firms affected
    """
    firms_failed_list = []
    costs_failed_list = []
    single_firm_list = []
    other_firm_list = []

    if group:
        for name_group in set(econ_sector.values()):
            index_group = firms_econ_sector(econ_sector, G, name_group)

            results = cascade_failures(A, theta, pnorm, index_group,
                                       drop_pct, failcost_pct)
            result_firms, result_costs, idx_failed_firms = results
            firms_failed_list.append(result_firms)
            costs_failed_list.append(result_costs)
            single_firm_list.append(name_group)
            other_firm_list.append([names[fidx] for fidx in idx_failed_firms])

    else:
        for f in np.arange(n):
            results = cascade_failures(A, theta, pnorm, [f],
                                       drop_pct, failcost_pct)
            result_firms, result_costs, idx_failed_firms = results
            firms_failed_list.append(result_firms)
            costs_failed_list.append(result_costs)
            single_firm_list.append(names[f])
            other_firm_list.append([names[fidx] for fidx in idx_failed_firms])

    df_single_firm = pd.DataFrame([single_firm_list, firms_failed_list,
                                   costs_failed_list, other_firm_list]).T
    df_single_firm.columns = ["Firms", "Total_Affected", "Total_Costs",
                              "List_Firms_Affected"]
    df = df_single_firm.sort_values("Total_Affected",
                                    ascending=False).reset_index(drop=True)
    return df


def firms_econ_sector(econ_sector, graph, name_group):
    """
    Create list of index of firms by economic sector
    """
    i = 0
    list_index = []
    for f in graph.nodes:
        if econ_sector[f] == name_group:
            list_index.append(i)
        i += 1
    return list_index
