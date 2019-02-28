# contrastive-LVM
Code for running and analyzing contrastive latent variable models (cLVMs). cLVM are described in K.A. Severson, S. Ghosh and K. Ng, 'Unsupervised learning with contrastive latent variable models' AAAI, 2019. Full details are available [here](https://arxiv.org/pdf/1811.06094.pdf). The cLVM model class is in `clvm_tfp.py`.

## Setting up the environment
clvm requires tensorflow-probability which is currently in developement. To ensure that the model works, please use the associated environment. To create the environment:

`conda env create -f tfp_env.yml`

Once the environment has been created, activate the environment:

`conda activate tfp_env`

The environment can be deactivated using:

`conda deactivate`

## Running the code

### Setting up the model
The clvm class requires:
1. A target dataset
2. A background dataset
3. The dimension of the shared latent space (default 10)
4. The dimension of the target latent space (default 2)

Note that the target and background datasets do not need to have the same number of observations but do need to have the same number of measurements per observation. cLVM will not automatically check for this.

Different versions of cLVMs can be used by setting certain flags to True:
* robust_flag: Inverse Gamma prior on the noise
* target_ARD: use ARD prior to perform model selection for the target space; note that the dimension of the target latent space 
 will not automatically be adjusted; choose a reasonable starting dimension
* shared_ARD: use ARD prior to perform modle selection for the shared space; not that the dimension of the shared space will not automatically be adjusted; choose a reasonable starting dimension
* target_missing: indicates missing values in the target dataset. Missing values should be represented by np.NaN
* background_missing: indicates missing values in the background dataset. Missing values should be represented by np.NaN

An example of a cLVM with shared spaced ARD and missing data in the test data would be:

`N, D = np.shape(x_train)`

`model = clvm(x_train, y_train, D-3, 2, shared_ARD=True, target_missing=True)`

### Performing inference
The clvm class has two inference methods built in:
1. MAP
2. VI

Both have the option to specify:
* num_epochs: number of iterations
* plot: flag for plotting the target latent space
* labels: labels for the plot
* seed: set a particular seed
* fn: filename to store the model results

Both methods return the target latent representation and save the learned model and objective function to a folder named results.
