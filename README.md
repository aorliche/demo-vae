# Demographic-Conditioned and Decorrelated Variational Autoencoder (DemoVAE)
Variational autoencoder generating synthetic subject FC and other fixed-length modalities of fMRI for distribution sampling, removal of confounds, and demographic-change or site harmonization.

# Installable pip package

Run 

```
pip install demovae
```

to install demovae along with basic dependencies (numpy, scikit-learn, and torch).

Check the file <a href='https://github.com/aorliche/demo-vae/blob/main/test/Pip3TestSample.ipynb'>Pip3TestSample.ipynb</a> for a basic idea of how to run the code.

Site harmonization may be achieved by fitting the model then transforming original data with changed "demographics" (i.e., site codes).

Check <a href='https://github.com/aorliche/demo-vae/blob/main/pip/src/demovae/sklearn.py'>this file</a> in the pip subdirectory to see all of the configuration parameters you can set, e.g.:

```
    @staticmethod
    def get_default_params():
        return dict(latent_dim=30,      # Latent dimension
                use_cuda=True,          # GPU acceleration
                nepochs=5000,           # Training epochs
                pperiod=100,            # Epochs between printing updates 
                bsize=1000,             # Batch size
                loss_C_mult=1,          # Covariance loss (KL div)
                loss_mu_mult=1,         # Mean loss (KL div)
                loss_rec_mult=1,        # Reconstruction loss
                loss_decor_mult=1,      # Latent-demographic decorrelation loss
                loss_pred_mult=0.001,   # Classifier/regressor guidance loss
                alpha=100,              # Regularization for continuous guidance models
                LR_C=100,               # Regularization for categorical guidance models
                lr=1e-4,                # Learning rate
                weight_decay=0,         # L2 regularization for VAE model
                )
```

# Features

- On-line demo available at <a href='https://aorliche.github.io/'>https://aorliche.github.io/DemoVAE/</a>
- Download the models from a link in the above app
- See server directory for how to use model to generate samples

We condition a variational autoencoder on demographic data, and at the same time train it to decorrelate latent
features from these demographics

<img src='https://github.com/aorliche/demo-vae/blob/3643570b74438692338e278cfcd541e69d20fb2c/image/Overview.png'>

This is important because many datasets are skewed with respect to demographics.

<img src='https://github.com/aorliche/demo-vae/blob/3643570b74438692338e278cfcd541e69d20fb2c/image/Demographics.png'>
<img src='https://github.com/aorliche/demo-vae/blob/3643570b74438692338e278cfcd541e69d20fb2c/image/WRAT.png'>

Prediction based on fMRI may be predicting the phenotype of interest OR predicting the demographic (age, sex, race, etc.) 
and inferring the phenotype based on the inherent demographic bias in the dataset.

In some cases, decorrelating latents from demographic information destroys predictive ability.

<img src='https://github.com/aorliche/demo-vae/blob/3643570b74438692338e278cfcd541e69d20fb2c/image/WRAT_Predict.png'>

We find that most clinical and computerized battery fields in the PNC and BSNIP datasets are biased by demographics,
and once latent features are decorrelated by DemoVAE, are no longer significantly correlated with the fMRI data.

<img src='https://github.com/aorliche/demo-vae/blob/3643570b74438692338e278cfcd541e69d20fb2c/image/Correlations.png'>

An exception to this finding are Antipsychotic medication use in the BSNIP dataset as well as PANSS scores for severity
of schizophrenia symptoms, which are still significantly correlated after demographic decorrelation.

We find the DemoVAE creates fMRI functional connectivity (FC) samples that are high in quality and represent the full distribution of subject fMRI.

<img src='https://github.com/aorliche/demo-vae/blob/3643570b74438692338e278cfcd541e69d20fb2c/image/Samples.png'><br>
<img src='https://github.com/aorliche/demo-vae/blob/3643570b74438692338e278cfcd541e69d20fb2c/image/tSNE.png'>

Additionally, it correctly captures the group differences of the demographic features it is trained on.

<img src='https://github.com/aorliche/demo-vae/blob/3643570b74438692338e278cfcd541e69d20fb2c/image/GroupDiff.png'>

Prediction using models trained on synthetic DemoVAE data is almost as good as prediction using models trained on real data.

<img src='https://github.com/aorliche/demo-vae/blob/3643570b74438692338e278cfcd541e69d20fb2c/image/Prediction.png'>

Preprint manuscript available at: <a href='https://arxiv.org/abs/2405.07977'>arXiv</a>.<br>
Submitted to IEEE journal.

Personal website: <a href='https://aorliche.github.io/'>https://aorliche.github.io/</a><br>
Lab website: <a href='https://www2.tulane.edu/~wyp/'>https://www2.tulane.edu/~wyp/</a><br>
<a href='mailto:aorlichenko@tulane.edu'>Email me</a>
