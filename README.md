## Towards systematic investigation of model transfer in Raman spectroscopy.

## Description
- Model transfer methods for Raman spectral analysis, including extensive multiplicative scattering correction (EMSC), score movement (MS), and Siamese network. 
- In addition, a generative netowrk based on variational autoencoder (VAE) is included to simulate spectra that share certain dissimilarity to an input spectrum. 
- Based on the simulated spectra, model transfer methods are characterized in terms of training sample size and the spectral variations between test and training data.

## Authors and acknowledgment
Shuxia Guo (Leibniz-IPHT, Jena), Thomas Bocklitz (Leibniz-IPHT, Jena; University Jena)

## Content
- code_mt_orgdata_k_batch.ipynb: test model transfer approaches with real data, in cases of different training sample size
- code_mt_gendata_k_batch.ipynb: characterize different model transfer in terms of different spectral variations and training sample size
- Classify.py: functions for PCA-LDA based classification and MS based model transfer
- modeltransfer.py: EMSC based model transfer
- SiameseNetwork.py: networks for network based model transfer: ordinary neural network, and siamese based network
- network_fcn.py: VAE networks for spectra generation
- utils.py: helping functions, including to prepare spectral pairs for VAE and siamese network.
- plot_model_transfer.py: plot accuracy results from validating and characterizing model transfer methods (Fig. 5, Fig. 7, and Fig. S2 of manuscript).
- plot_others.py: plot other results from intermediate calculations (Fig. 6 and Fig. S1 of manuscript).
- /datasets: dataset used in the manuscript
- /results: accuracy results



