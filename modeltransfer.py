import numpy as np
from sklearn.decomposition import PCA

def emsc(x, degree=2, reference=None, interferent=None, constituent=None, wn=None, interf_pca=0, contit_pca=0):
    """
    Extended multiplicative scatter correction (EMSC) model-based preprocessing.

    :param x: ndarray, or list with spectroscopic data
    :param degree: polynomial degree used to fit the baseline
    :param reference: reference spectrum (mean over the data set if None)
    :param interferent: a dict with components to be substracted (None by default)
    :param constituent: a dict with components to preserve (None by default)
    :param wn: wavenumber axis
    :param rep_pca: used number of components for PCA in interference (0 by default); if <1, interference used as is
    :return: SpData object, ndarray, or list (depending on input) with corrected spectra
    """
    if reference is None or not len(reference):
        reference = np.asarray(x).mean(axis=0)
        reference = (reference - reference.min()) / reference.ptp()
    if wn is None or not len(wn):
        wn = np.arange(len(x[0]))
    reference = np.asarray(reference)

    # Step 1 (prepare polynomials): get a matrix with orthogonal polynomials up to a needed degree
    wn_len = wn.ptp()
    wn_mean = wn.mean()
    mat_orth = np.asarray([(2 * (wn_mean - wn) / wn_len) ** d for d in range(degree + 1)])
    mat_orth = np.linalg.qr(mat_orth.T)[0].T[:degree + 1]
    mat_orth[0] = np.abs(mat_orth[0])

    # Step 2 (model): combine reference, matrix with polynomials, interferent, and constituent spectra
    mat_full = [reference, mat_orth]
    if interferent is not None:
        interf_mat = np.asarray(interferent)
        if interf_pca:
            if interf_mat.shape[0] > 1:
                interf_mat = emsc(interf_mat, degree=degree)['corrected']
                interf_mat = PCA(int(min([*interf_mat.shape, interf_pca]))).fit(interf_mat).components_
            else:
                interf_mat = emsc(interf_mat, degree=degree, reference=reference)['corrected'] - reference
        mat_full.append(interf_mat)
        len_interf = interf_mat.shape[0]
    else:
        len_interf = 0

    if constituent is not None:
        contit_mat = np.asarray(constituent)
        if contit_pca:
            if contit_mat.shape[0] > 1:
                contit_mat = emsc(contit_mat, degree=degree)['corrected']
                contit_mat = PCA(int(min([*contit_mat.shape, contit_pca]))).fit(contit_mat).components_
            else:
                contit_mat = emsc(contit_mat, degree=degree, reference=reference)['corrected'] - reference
        mat_full.append(contit_mat)
        
    mat_full = np.vstack(mat_full)

    # Step 3: apply model to each spectrum
    w_rows = range(1, degree + len_interf + 2)  # rows with polynomials and interferent spectra only

    def emsc_single(_x):
        coef = np.linalg.lstsq(mat_full.T, _x, rcond=-1)[0]
        bg = np.dot(coef[w_rows], mat_full[w_rows])
        return (_x-bg)/(coef[0] or 1), bg.mean()  # returns corrected spectrum and background intensity
    res = [emsc_single(xi) for xi in x]

    # format output
    res = {
        "corrected": np.asarray([r[0] for r in res]),
        "bg_area": np.asarray([r[1] for r in res]),
        "reference": reference.tolist(),
        "interferent": interferent,
        "constituent": constituent
    }
    return res

