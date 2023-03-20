import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_mets_data(fts_ctg, fts = [], gender=None, ONEHOT=False, TABNET=False, mrmr=True, SEED=100):
    
    # Path
    label = 'mets'
    PATH = './data/'

    dpath = PATH + 'mets_data.csv'
    cpath = PATH + 'col_info.csv'
    additional_info = ['rid','testset','mets_num','RMRS','RMRS_type']

    # Load
    mets_ds = pd.read_csv(dpath)
    col_info = pd.read_csv(cpath)
    
    # For TabNet : categorical variable one_hot encoding
    #one_hot = True    
    if ONEHOT :

        ctg_lg = col_info.onehot == 'Y'
        ctg_col = col_info.col_name[ctg_lg]
        ctg_idx = mets_ds.columns.isin(ctg_col)

        oh_fts = pd.get_dummies(mets_ds.loc[:,ctg_idx].astype(str))
        mets_ds = pd.concat([oh_fts, mets_ds.drop(columns = mets_ds.columns[ctg_idx])], axis=1)

        atb = []
        ctg = []
        typ = []
        ctg_col_extnd = oh_fts.columns

        for i, name in enumerate(ctg_col_extnd):
            pos = name.rfind('_')
            idx = (col_info.col_name == name[0:pos])
            atb.extend(col_info.loc[idx,'attribute'].values.tolist())
            ctg.extend(col_info.loc[idx,'categorical'].values.tolist())
            typ.extend(col_info.loc[idx,'type'].values.tolist())

        ctg_info = pd.DataFrame({'col_name':ctg_col_extnd, 'categorical': ctg,'attribute':atb, 'type': typ})
        idx = ~(col_info.onehot == 'Y') & ~(col_info.attribute == 'I')
        col_info = pd.concat([ctg_info, col_info.loc[idx,['col_name','categorical','attribute','type']]], axis=0)
        
    # Spilit data into training and test data 
    info = mets_ds[additional_info]
    tr = mets_ds.loc[mets_ds.testset == 0, :].drop(columns = (additional_info))
    ts = mets_ds.loc[mets_ds.testset == 1, :].drop(columns = (additional_info))
    
    if gender is not None:
        tr = tr.loc[tr.sex == gender,:]
        ts = ts.loc[ts.sex == gender,:]
    
    tr, val = train_test_split(tr, test_size=0.1, random_state=SEED)
    
    # For Undersampling 
    pos_idx = tr[label]==1
    n_all = tr[label].size
    n_pos = tr.loc[pos_idx,label].size
    n_neg = tr.loc[~pos_idx,label].size

    beta = n_pos/n_neg
    tau = n_pos/n_all
    
    # Downsampling
    neg = tr.loc[~pos_idx,:]
    pos = tr.loc[pos_idx,:]

    np.random.seed(SEED)
    down_neg_idx = np.random.choice(neg.shape[0], size=pos.shape[0], replace=False)

    down_neg = neg.iloc[down_neg_idx,:]
    tr = pd.concat([pos,down_neg])
    
    # Set Features
    idx = []
    if fts_ctg == 'all' :
        idx = (col_info.categorical == fts_ctg)
        fts = mets_ds.columns.to_list()
    else :
        idx = (col_info.categorical == fts_ctg)
        fts.extend(col_info.col_name[idx].to_list())
        
    if mrmr :
        if gender == 0 :
            mrmr_fts = pd.read_csv(PATH+'mrmr_male.csv')
        elif gender == 1 :
            mrmr_fts = pd.read_csv(PATH+'mrmr_female.csv')
        else :
            mrmr_fts = pd.read_csv(PATH+'mrmr_all.csv')
        fts = list(set(fts) & set(mrmr_fts['features']))
    
    # Dataset
    sets = [label]+fts
    tr = tr.loc[:,sets]
    val = val.loc[:,sets]
    ts = ts.loc[:,sets]
    
    # For tabnet : categorical info
    cat_col = col_info.col_name[(col_info.type == 'C') & ~(col_info.attribute == 'I')]

    base_col = tr.columns.drop([label])
    cat_cols = tr.columns[tr.columns.isin(cat_col)]

    ctg_dim = {}
    ctg_idx = {}

    if TABNET :
        # For TabNet
        for col in cat_cols :
            ctg_dim[col] = tr.loc[:,col].nunique()

        ctg_idx = [ i for i, f in enumerate(base_col) if f in cat_cols]
        ctg_dim = [ ctg_dim[f] for i, f in enumerate(base_col) if f in cat_cols]
        
    return tr, val, ts, info, beta, tau, ctg_idx, ctg_dim