# Demographic-Conditioned and Decorrelated Variational Autoencoder (DemoVAE)
Variational autoencoder generating synthetic subject FC and timeseries based on various datasets (PNC and BSNIP)

- Models available in the data directory
- See server directory for how to use model to generate samples
- On-line demo available at <a href='https://aorliche.github.io/'>https://aorliche.github.io/?post=generative-fc</a>

We condition a variational autoencoder on demographic data, and at the same time train it to decorrelate latent
features from these demographics

This is important because many datasets are skewed with respect to demographics.

Prediction based on fMRI may be predicting the phenotype of interest OR predicting the demographic (age, sex, race, etc.) 
and inferring the phenotype based on the inherent demographic bias in the dataset.

We find that most clinical and cognitive battery fields in the PNC and BSNIP datasets are biased by demographics,
and once latent features are decorrelated by DemoVAE, are no longer correlated with the fMRI data.

An exception to this finding are Antipsychotic medication use in the BSNIP dataset as well as PANSS scores for severity
of schizophrenia symptoms, which are still significantly correlated after demographic decorrelation.

We find the DemoVAE creates fMRI functional connectivity (FC) samples that are high in quality and represent the full distribution of subject fMRI.

Additionally, it correctly captures the group differences of the demographic features it is trained on.

Prediction using models trained on synthetic DemoVAE data is almost as good as prediction using models trained on real data.

Preprint manuscript available at: (coming soon).
Submitted to IEEE journal.

Personal website: <a href='https://aorliche.github.io/'>https://aorliche.github.io/</a>
Lab website: <a href='https://www2.tulane.edu/~wyp/'>https://www2.tulane.edu/~wyp/</a>
<a href='mailto:aorlichenko@tulane.edu'>Email me</a>
