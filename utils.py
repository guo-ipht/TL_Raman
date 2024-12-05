import numpy as np
from scipy.signal import savgol_filter
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as mcolors
from itertools import combinations
from scipy import spatial

def snip(spectra, snip_iter, snip_w=11):
    bg = savgol_filter(spectra, snip_w, 2) if snip_w > 2 else spectra.copy()
    bg[bg < 0] = 0
    bg = np.log(np.log(np.sqrt(bg + 1) + 1) + 1)  # log-log-square_root (LLS) operator
    for p in range(1, snip_iter + 1, 1):  # optimized snip loop (about 12 times faster)
        bg[:, p:-p] = np.minimum(bg[:, p:-p], (bg[:, p * 2:] + bg[:, :-p * 2]) / 2)
    bg = (np.exp(np.exp(bg) - 1) - 1) ** 2 - 1  # back transformation of LLS operator
    spectra = spectra - bg
    return spectra

def prep_training(spec, labels, batches, n_pairs=20000):
    index = np.array(range(spec.shape[0]))
    x = []; y=[]; c=[]; g=[];  
    for k in range(spec.shape[0]):
        x.append(spec[k, :])
        y.append(spec[k, :])
        g_tmp = np.zeros([6])
        g_tmp[:2] = [1.0, 1.0]
        g_tmp[2:] = np.asarray([np.random.uniform(0.9, 1) for ii in range(4)])
        g.append(g_tmp[np.newaxis, :])
        
    comb = list(combinations(index, 2))
    ks = range(0, len(comb), np.max([1,len(comb)//(3*n_pairs)]))
    for k in range(len(comb)):
        x.append(spec[comb[k][0], :])
        y.append(spec[comb[k][1], :])
        c0 = 1-spatial.distance.cosine(spec[comb[k][0], :], spec[comb[k][1], :]) 
        c1 = np.corrcoef(spec[comb[k][0], :], spec[comb[k][1], :])[0, 1]
        
        g_tmp = np.zeros([6])
        g_tmp[:2] = [c0**8, c1**8]
        
        if labels[comb[k][0]] != labels[comb[k][1]]:   ### different groups
            g_tmp[2:] = np.asarray([np.random.uniform(0.01, 0.1) for ii in range(4)])
            g.append(g_tmp[np.newaxis, :])
        elif batches[comb[k][0]] == batches[comb[k][1]]:  ### same group, same batch
            g_tmp[2:] = np.asarray([np.random.uniform(0.9, 1) for ii in range(4)])
            g.append(g_tmp[np.newaxis, :])
        else:  ### same group, different batch
            bb = np.asarray([np.random.uniform(0.01, 0.1) for ii in range(2)])
            gg = np.asarray([np.random.uniform(0.9, 1) for ii in range(2)])
            g_tmp[2:] = np.append(bb, gg) 
            g.append(g_tmp[np.newaxis, :])
            
    x = np.row_stack(x)
    y = np.row_stack(y)
    g = np.concatenate(g, axis=0)

    ix_dg = np.argwhere(g[:, -1]<0.5)[:,0] ### different group
    ix_sg_db = np.argwhere((g[:, 2]<0.5) & (g[:, -1]>0.5))[:,0] ### same group different batch
    ix_sg_sb = np.argwhere((g[:, 2]>0.5) & (g[:, -1]>0.5))[:,0] ### same group same batch

    ix_sel = []
    cors = np.linspace(np.min(g[ix_dg, 0]), 1, n_pairs//2)
    for c in cors:
        ix_sel = np.append(ix_sel, ix_dg[np.argmin(np.abs(g[ix_dg, 0]-c))])

    cors = np.linspace(np.min(g[ix_sg_db, 0]), 1, n_pairs//4)
    for c in cors:
        ix_sel = np.append(ix_sel, ix_sg_db[np.argmin(np.abs(g[ix_sg_db, 0]-c))])

    cors = np.linspace(np.min(g[ix_sg_sb, 0]), 1, n_pairs//4)
    for c in cors:
        ix_sel = np.append(ix_sel, ix_sg_sb[np.argmin(np.abs(g[ix_sg_sb, 0]-c))])
    
    ix_sel = np.array(ix_sel, dtype='int64')
    x = x[ix_sel,:]
    y = y[ix_sel,:]
    g = g[ix_sel,:]

    arr = np.array(range(x.shape[0]))
    np.random.shuffle(arr)
    x = x[arr,:]
    y = y[arr,:]
    g = g[arr,:]
    
    return x,y,g

    
def prep_training_cls(spec, labels, batches, n_pairs=10000, cb_shape=10, n_sep=(2, 4, 4)):
    spec_in = []; spec_out=[]; cb_in_out=[]; g_in_out=[]; c_in_out=[]
    index = np.array(range(spec.shape[0]))
    comb = list(combinations(index, 2))
    ks = range(0, len(comb), np.max([1,len(comb)//(3*n_pairs)]))
    for k in ks: 
        spec_in.append(spec[comb[k][0], :])
        spec_out.append(spec[comb[k][1], :])
        c_in_out = np.append(c_in_out, 0.5*(1-spatial.distance.cosine(spec[comb[k][0], :], spec[comb[k][1], :]) + np.corrcoef(spec[comb[k][0], :], spec[comb[k][1], :])[0, 1]))
        g_tmp = [] 
        
        if labels[comb[k][0]] != labels[comb[k][1]]:   ### different groups, different batch
            g_tmp = np.append(g_tmp, np.asarray([np.random.uniform(0.01, 0.1) for ii in range(cb_shape)]))
            g_in_out = np.append(g_in_out, np.random.uniform(0.01, 0.1))
        elif batches[comb[k][0]] == batches[comb[k][1]]:  ### same group, same batch
            g_tmp = np.append(g_tmp, np.asarray([np.random.uniform(0.9, 1) for ii in range(cb_shape)]))
            g_in_out = np.append(g_in_out, np.random.uniform(0.9, 1))
        else:  ### same group, different batch
            g_tmp = np.append(g_tmp, np.asarray([np.random.uniform(0.01, 0.1) for ii in range(cb_shape)]))
            g_in_out = np.append(g_in_out, np.random.uniform(0.9, 1))
        cb_in_out.append(g_tmp)

    spec_in = np.row_stack(spec_in)
    spec_out = np.row_stack(spec_out)
    cb_in_out = np.row_stack(cb_in_out)

    ix_dg = np.argwhere(g_in_out<0.5)[:,0] ### different group
    ix_sg_db = np.argwhere((g_in_out>0.5) & (cb_in_out[:, 0]<0.5))[:,0] ### same group different batch
    ix_sg_sb = np.argwhere((g_in_out>0.5) & (cb_in_out[:, 0]>0.5))[:,0] ### same group same batch

    ix_sel = []
    cors = np.linspace(np.min(c_in_out[ix_dg]), 1, n_pairs//n_sep[0])
    for c in cors:
        ix_sel = np.append(ix_sel, ix_dg[np.argmin(np.abs(c_in_out[ix_dg]-c))])

    cors = np.linspace(np.min(c_in_out[ix_sg_db]), 1, n_pairs//n_sep[1])
    for c in cors:
        ix_sel = np.append(ix_sel, ix_sg_db[np.argmin(np.abs(c_in_out[ix_sg_db]-c))])

    cors = np.linspace(np.min(c_in_out[ix_sg_sb]), 1, n_pairs//n_sep[2])
    for c in cors:
        ix_sel = np.append(ix_sel, ix_sg_sb[np.argmin(np.abs(c_in_out[ix_sg_sb]-c))])

    ix_sel = np.array(ix_sel, dtype='int64')
    spec_in = spec_in[ix_sel,:]
    spec_out = spec_out[ix_sel,:]
    cb_in_out = cb_in_out[ix_sel,:]
    g_in_out = g_in_out[ix_sel]
    
    arr = np.array(range(spec_in.shape[0]))
    np.random.shuffle(arr)
    spec_in = spec_in[arr,:]
    spec_out = spec_out[arr,:]
    cb_in_out = cb_in_out[arr, :]
    g_in_out = g_in_out[arr]

    return spec_in, spec_out, cb_in_out, g_in_out

def prep_training_cls2(spec1, labels1, batches1, spec2, labels2, batches2, n_pairs=10000, cb_shape=10):
    spec_in = []; spec_out=[]; cb_in_out=[]; g_in_out=[]; 
    index1 = np.array(np.random.choice(range(spec1.shape[0]), n_pairs, replace=True))
    index2 = np.array(np.random.choice(range(spec2.shape[0]), n_pairs, replace=True))
    for k in range(n_pairs):
        spec_in.append(spec1[index1[k], :])
        spec_out.append(spec2[index2[k], :])
        g_tmp = [] 
        
        if labels1[index1[k]] != labels2[index2[k]]:   ### different groups, different batch
            g_tmp = np.append(g_tmp, np.asarray([np.random.uniform(0.01, 0.1) for ii in range(cb_shape)]))
            g_in_out = np.append(g_in_out, np.random.uniform(0.01, 0.1))
        elif batches1[index1[k]] == batches2[index2[k]]:  ### same group, same batch
            g_tmp = np.append(g_tmp, np.asarray([np.random.uniform(0.9, 1) for ii in range(cb_shape)]))
            g_in_out = np.append(g_in_out, np.random.uniform(0.9, 1))
        else:  ### same group, different batch
            g_tmp = np.append(g_tmp, np.asarray([np.random.uniform(0.01, 0.1) for ii in range(cb_shape)]))
            g_in_out = np.append(g_in_out, np.random.uniform(0.9, 1))
        cb_in_out.append(g_tmp)

    spec_in = np.row_stack(spec_in)
    spec_out = np.row_stack(spec_out)
    cb_in_out = np.row_stack(cb_in_out)
    
    arr = np.array(range(spec_in.shape[0]))
    np.random.shuffle(arr)
    spec_in = spec_in[arr,:]
    spec_out = spec_out[arr,:]
    cb_in_out = cb_in_out[arr, :]
    g_in_out = g_in_out[arr]

    return spec_in, spec_out, cb_in_out, g_in_out
    
def do_PCA_LDA(spec, labels, batches, ix, ix_test, nPCs=range(3, 50), ix_wn=None):
    acc = []
    min_acc = []
    std_acc = []
    method = []
    b_test = []
    if ix_wn is None:
        ix_wn = np.array(range(spec.shape[1]))
    for npc in nPCs:
        m_pca = PCA(n_components=npc).fit(spec[ix,:][:, ix_wn])
        scr_train = m_pca.transform(spec[ix,:][:, ix_wn])
        scr_test = m_pca.transform(spec[ix_test,:][:, ix_wn])
        m_lda = LDA().fit(scr_train, labels[ix])
        pred = m_lda.predict(scr_test)
        for ll in np.unique(batches[ix_test]):
            i_b = np.argwhere(batches[ix_test]==ll)[:,0]
            cc = confusion_matrix(labels[ix_test][i_b], pred[i_b], labels=np.unique(labels))
            accs_all = np.diag(cc)/np.sum(cc, 1)
            min_acc = np.append(min_acc, np.nanmin(accs_all))
            std_acc = np.append(std_acc, np.nanstd(accs_all))
            acc = np.append(acc, np.nanmean(accs_all))
            b_test = np.append(b_test, ll)
            method = np.append(method, npc)
    return acc, min_acc, std_acc, b_test, method

def SRD(mat, reference=None, type=['maximum', 'minimum', 'mean', 'median'][0]):
    if reference is None:
        if type == 'maximum': 
            reference = np.max(mat, axis=1)
        elif type == 'minimum':
            reference = np.min(mat, axis=1)
        elif type == 'mean':
            reference = np.mean(mat, axis=1)
        elif type == 'median':
            reference = np.median(mat, axis=1)
        else:
            reference = np.mean(mat, axis=1)
    ix = np.argsort(reference)
    reference = reference[ix]
    mat1 = mat[ix,:]
    val = []
    for i in range(mat1.shape[1]):
        dTmp2 = np.argsort(mat1[:,i])
        val = np.append(val, np.sum(abs(mat1[dTmp2,i]-reference)))
    return val