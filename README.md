# contrastive-LVM
Code for running and analyzing contrastive latent variable models (cLVMs). cLVM are described in K.A. Severson, S. Ghosh and K. Ng, 'Unsupervised learning with contrastive latent variable models' AAAI, 2019. Full details are available [here](https://arxiv.org/pdf/1811.06094.pdf). The cLVM model class is in `clvm_tfp.py`.

## Setting up the environment
clvm requires tensorflow-probability which is currently in development. To ensure that the model works, please use the associated environment. To create the environment:

`conda env create -f tfp_env.yml`

Once the environment has been created, activate the environment:

`conda activate tfp_env`

The environment can be deactivated using:

`conda deactivate`

## Running the code
See the jupyter notebook cLVM Sample Code in the experiments folder for a simple example of the model.

### Setting up the model
The clvm class requires:
1. A target dataset, with N rows of observations, each with D measurements
2. A background dataset, with M rows of observations, each with D measurements
3. The dimension of the shared latent space (default 10)
4. The dimension of the target latent space (default 2)

Note that the target and background datasets do not need to have the same number of observations but do need to have the same number of measurements per observation. A warning will be printed if this is not true and inference will fail.

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
To perform inference on the clvm, use the variational inference function.

VI has the follwoing options to specify:
* num_epochs: number of iterations (default 10000)
* plot: flag for plotting the target latent space (default False)
* labels: labels for the plot (default None)
* seed: set a particular seed (default 1234)
* fn: filename to store the model results (default 'model_MAP'/'model_VI')
* fp: filepath to store the model results (default './results')
* saveGaph: flag for saving the tesnorflow graph (default False)
* paramsOnly: flag to indicate if all of the variables in the graph should be saved or only the parameters;i.e. variables that are not a function of n (default True)

VI returns the target latent representation and save the learned model and objective function to a folder named results. An example of variational inference applied to the above model would be:

`t_hat = model.variational_inference(num_epochs=5000, fn='trained model', fp='../results/')`

### Generative sampling
Once a model has been trained, it is possible to generate samples using the posterior estimates. 

`target_gen, background_gen = model.generate()`

Currently `generate` has only one option: 
* use_inferred: boolean to indicate if the inferred latent variables should be used, default=True. If false, samples are drawn from the latent variable prior distribution.

### Saving and restoring the graph
You can choose the save the full graph or only the parameters of the graph by specifying `saveGraph=True` and choosing the setting for `paramsOnly`. `restore_graph` assumes that you are restoring the graph to continue training. If the training process is continuing with a different dataset, use `paramsOnly=True`. To restore the graph use

`model.restore_graph(fl='.checkpoint/model1234.ckpt')`

restore_graph has the options:
* fl: checkpoint file to load with the saved graph
plot: flag for plotting the target latent space (default False)
* labels: labels for the plot (default None)
* seed: set a particular seed (default 1234)
* fn: filename to store the model results (default 'model_MAP'/'model_VI')
* fp: filepath to store the model results (default './results')
* saveGaph: flag for saving the tesnorflow graph (default False)
* paramsOnly: flag to indicate if all of the variables in the graph should be saved or only the parameters;i.e. variables that are not a function of n (default True)

### Applying a trained cLVM model to test data
Applying a trained cLVM model to new data uses a different class found in `apply_clvm_tfp.py`. The `apply_clvm` class requires:
1. A model pkl for the learned cLVM
2. A target dataset, with N rows of observations, each with D measurements. D must be the same as the dimensionality of the data used to train the model but N is unrestricted.

A background dataset can also be optional supplied. Model flags and dimensionality will be loaded from the model pkl automatically. Missing data flags should be specified to reflect the testing data.

Inference follows as before, with the same options.
