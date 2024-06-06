
from demovae.model import VAE, train_vae, to_numpy, demo_to_torch

from sklearn.base import BaseEstimator

class DemoVAE(BaseEstimator):
    def __init__(self, **params):
        self.set_params(**params)

    @staticmethod
    def get_default_params():
        return dict(latent_dim=100, 
                use_cuda=True,
                nepochs=5000,           # Training epochs
                pperiod=100,            # Epochs between printing updates 
                bsize=1000,             # Batch size
                loss_C_mult=1,          # Covariance loss (KL div)
                loss_mu_mult=1,         # Mean loss (KL div)
                loss_rec_mult=10,       # Reconstruction loss
                loss_decor_mult=1,      # Latent-demo decorrelation loss
                loss_pred_mult=0.001    # Classifier/regressor guidance loss
                )

    def get_params(self, **params):
        return dict(latent_dim=self.latent_dim,
                use_cuda=self.use_cuda,
                nepochs=self.nepochs,
                pperiod=self.pperiod,
                bsize=self.bsize,
                loss_C_mult=self.loss_C_mult,
                loss_mu_mult=self.loss_mu_mult,
                loss_rec_mult=self.loss_rec_mult,
                loss_decor_mult=self.loss_decor_mult,
                loss_pred_mult=self.loss_pred_mult
                )

    def set_params(self, **params):
        dft = DemoVAE.get_default_params()
        for key in dft:
            if key in params:
                setattr(self, key, params[key])
            else:
                setattr(self, key, dft[key])
        return self

    def fit(self, x, demo, demo_types, **kwargs):
        # Get demo_dim
        demo_dim = 0
        for d,t in zip(demo, demo_types):
            if t == 'continuous':
                demo_dim += 1
            elif t == 'categorical':
                ll = len(set(list(d)))
                if ll == 1:
                    print('Only one type of category for categorical variable')
                    raise Exception('Bad categorical')
                demo_dim += ll
            else:
                print(f'demographic type "{t}" not "continuous" or "categorical"')
                raise Exception('Bad demographic type')
        # Create model
        self.vae = VAE(x.shape[1], self.latent_dim, demo_dim, self.use_cuda)
        # Train model
        self.pred_stats = train_vae(vae, x, demo, demo_types, 
                self.nepochs, self.pperiod, self.bsize, 
                self.loss_C_mult, self.loss_mu_mult, self.loss_rec_mult, self.loss_decor_mult, self.loss_pred_mult)

    def transform(self, x, demo, demo_types, **kwargs):
        if isinstance(x, int):
            # Generate
            z = self.vae.gen(x)
        else:
            # Get latents for real data
            z = self.vae.enc(x)
        demo_t = demo_to_torch(demo, demo_types, self.pred_stats)
        y = self.vae.dec(x, demo_t)
        return to_numpy(y)

    def fit_transform(self, x, demo, demo_types, **kwargs):
        self.fit(x, demo, demo_types)
        return self.transform(x, demo, demo_types)

