# -*- coding: utf-8 -*-

import pandas as pd

def physics_train(df,full_cleaning=True):
    df=df.drop(columns=['PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'DER_lep_eta_centrality', 'DER_prodeta_jet_jet', 'DER_mass_jet_jet', 'DER_deltaeta_jet_jet','PRI_jet_leading_pt','PRI_jet_leading_eta','PRI_jet_leading_phi'])
    if full_cleaning:
        df=df.drop(columns=['PRI_jet_all_pt', 'DER_sum_pt', 'PRI_met_sumet', 'DER_pt_h'])
    median=np.median(df['DER_mass_MMC'])
    df['DER_mass_MMC'][df['DER_mass_MMC']==-999.0]=median
    return df ,median


def physics_test(df_test, median, full_cleaning=True):
    df_test=df_test.drop(columns=['PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'DER_lep_eta_centrality', 'DER_prodeta_jet_jet', 'DER_mass_jet_jet', 'DER_deltaeta_jet_jet','PRI_jet_leading_pt','PRI_jet_leading_eta','PRI_jet_leading_phi'])
    if full_cleaning:
        df_test=df_test.drop(columns=['PRI_jet_all_pt', 'DER_sum_pt', 'PRI_met_sumet', 'DER_pt_h'])
    df_test['DER_mass_MMC'][df_test['DER_mass_MMC']==-999.0]=median
    return df_test





