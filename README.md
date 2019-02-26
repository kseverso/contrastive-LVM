# contrastive-LVM
Code for running and analyzing contrastive latent variable models (cLVMs). cLVM are described in K.A. Severson, S. Ghosh and K. Ng, 'Unsupervised learning with contrastive latent variable models' AAAI, 2019. Full details are available [here](https://arxiv.org/pdf/1811.06094.pdf)

## Setting up the environment

## Running the code
The clvm class requires:
1. A target dataset
2. A background dataset
3. The dimension of the shared latent space (default 10)
4. The dimension of the target latent space (default 2)

Different versions of cLVMs can be used by setting certain flags to True:
* robust_flag: Inverse Gamma prior on the noise
* target_ARD: use ARD prior to perform model selection for the target space; note that the dimension of the target latent space 
 will not automatically be adjusted; choose a reasonable starting dimension
* shared_ARD: use ARD prior to perform modle selection for the shared space; not that the dimension of the shared space will not automatically be adjusted; choose a reasonable starting dimension
* target_missing: indicates missing values in the target dataset. Missing values should be represented by np.NaN
* background_missing: indicates missing values in the background dataset. Missing values should be represented by np.NaN
