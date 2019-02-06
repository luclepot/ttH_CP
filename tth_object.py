from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from madminer.core import MadMiner
from madminer.plotting import plot_2d_morphing_basis
from madminer.delphes import DelphesProcessor
from madminer.sampling import combine_and_shuffle
from madminer.sampling import SampleAugmenter
from madminer.sampling import constant_benchmark_theta, multiple_benchmark_thetas
from madminer.sampling import constant_morphing_theta, multiple_morphing_thetas, random_morphing_thetas
from madminer.ml import MLForge
import logging
import corner

from time import time


def new_tth_object(samples, theta_samples=100): 
    t = tth(n_samples=samples)
    t.sampling.init()
    t.sampling.train_ratio(n_theta_samples=theta_samples)
    t.sampling.extract_samples()
    return t

def train_and_evaluate(t, name="carl", node_arch=(20,20,20), n_epochs=10, batch_size=512):
    t.training.init()
    t.training.train_method(name=name, node_arch=node_arch, n_epochs=n_epochs, batch_size=batch_size)
    t.validation.run()
    return t

class tth_util:
    def _tprint(self, *inputs):
        for inp in inputs:
            print(self._indent(text=inp, amount=1, ch='    > '))

    def _hprint(self, *inputs):
        for inp in inputs:
            print(self._header(text=inp, amount=3, ch='-'))

    def _header(self, text, amount, ch=' '):
        padding = amount * ch
        return "".join(padding + " " + line + " " + padding for line in text.splitlines(True))

    def _indent(self, text, amount, ch=' '):
        padding = amount * ch
        return ''.join(padding+line for line in text.splitlines(True))

class sampling(tth_util):

    def __init__(self, use_parton_level=True, n_samples=100000):
        self.use_parton_level = use_parton_level
        self.n_samples = n_samples

    def run(self): 
        self._hprint("Starting sampling run")
        self._tprint("", "Initializing")
        self.init()
        self._tprint("", "Training ratio")
        self.train_ratio()
        self._tprint("", "Extracting samples", "")
        self.extract_samples()
        self._hprint("Finished sampling run")
        # self.plot_distributions()

    def init(self):
        if self.use_parton_level:
            self.sa = SampleAugmenter('data/madminer_example_shuffled_parton.h5')
        else:
            self.sa = SampleAugmenter('data/madminer_example_shuffled_reco.h5')
        
    def train_ratio(self, n_theta_samples=100):
        self.x, self.theta0, self.theta1, self.y, self.r_xz, self.t_xz = self.sa.extract_samples_train_ratio(
            theta0=random_morphing_thetas(n_theta_samples, [('flat', 0., 1.)]),
            theta1=constant_benchmark_theta('sm'),
            n_samples=self.n_samples,
            folder='./data/samples',
            filename='train1'
        )

    def extract_samples(self):
        self.x, self.theta = self.sa.extract_samples_test(
            theta=constant_benchmark_theta('sm'),
            n_samples=self.n_samples,
            folder='./data/samples',
            filename='test'
        )

        self.x_bsm, self.theta_bsm = self.sa.extract_samples_test(
            theta=constant_benchmark_theta('w'),
            n_samples=self.n_samples,
            folder='./data/samples',
            filename='test_bsm'
        )

        self.x_bsm_morph, self.theta_bsm_morph = self.sa.extract_samples_test(
            theta=constant_benchmark_theta('morphing_basis_vector_2'),
            n_samples=self.n_samples,
            folder='./data/samples',
            filename='test_bsm_morph'
        )

    def plot_distributions(self):
        if self.use_parton_level: # parton level analysis
            labels = [r'$\Delta\eta_{t,\bar{t}}$', r'$p_{T, x0}$ [GeV]']
            ranges = [(-8., 8.), (0., 600.)]
            bins   = (25,25)
        else:
            labels = [r'$\Delta \phi_{\gamma \gamma}$', r'$p_{T, \gamma \gamma}$']
            bins   = (25,25)
            ranges = [(-3.15, 3.15), (0., 600.)]

        fig = corner.corner(self.x_bsm_morph, color='C2', labels=labels, range=ranges, bins=bins)
        _ = corner.corner(self.x_bsm, color='C1', labels=labels, range=ranges, bins=bins, fig=fig)
        _ = corner.corner(self.x, color='C0', labels=labels, range=ranges, bins=bins, fig=fig)
        fig.show()
    
class training(tth_util):
    def __init__(self):
        # MadMiner output
        logging.basicConfig(
            format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
            datefmt='%H:%M',
            level=logging.INFO
        )

        # Output of all other modules (e.g. matplotlib)
        for key in logging.Logger.manager.loggerDict:
            if "madminer" not in key:
                logging.getLogger(key).setLevel(logging.WARNING)
    
    def run(self):
        self._hprint("Starting training run")
        self._tprint("", "Initializing")
        self.init()
        self._tprint("", "Training methods")
        self.train_method()
        self._tprint("")
        self._hprint("Finished training run")

    def init(self):
        self.forge = MLForge()

    def train_method(self,
                     name='alice',
                     node_arch=(20,20,20),
                     n_epochs=20,
                     batch_size=512):
        self.forge.train(
            method=name,
            theta0_filename='data/samples/theta0_train1.npy',
            x_filename='data/samples/x_train1.npy',
            y_filename='data/samples/y_train1.npy',
            r_xz_filename='data/samples/r_xz_train1.npy',
            t_xz0_filename='data/samples/t_xz_train1.npy',
            n_hidden=node_arch,
        #    alpha=5.,
        #    initial_lr=0.1,
            n_epochs=n_epochs,
            validation_split=0.3,
            batch_size=batch_size
        )
        self.forge.save('models/model')

class validation(tth_util):
    has_init = False
    has_evaluate = False

    def __init__(self): 
        pass

    def run(self):
        self._hprint("Starting validation run")
        self._tprint(" ", "Initalizing")
        self.init(grid_spacing=21)
        self._tprint(" ", "Evaluating data")
        self.evaluate()
        self._tprint(" ", "Plotting results")
        self.plot_results()
        self._hprint("Finished validation run")

    def init(self, grid_spacing=21):
        self.theta_each = np.linspace(0.,1.,grid_spacing)
        #theta0, theta1 = np.meshgrid(theta_each, theta_each)
        #theta_grid = np.vstack((theta0.flatten(), theta1.flatten())).T
        self.theta_grid = np.array([self.theta_each]).T
        np.save('data/samples/theta_grid.npy', self.theta_grid)

        self.theta_denom = np.array([[0.5,0.5]])
        np.save('data/samples/theta_ref.npy', self.theta_denom)
        self.has_init = True

    def evaluate(self):
        if not self.has_init:
            self._tprint("QUITTING: validation has not initialized")
            return 
        self.forge = MLForge()
        self.forge.load('models/model')
        self._tprint("Evaluating x_test:",)
        t0 = time() 
        self.log_r_hat, _, _ = self.forge.evaluate(
            theta0_filename='data/samples/theta_grid.npy',
            x='data/samples/x_test.npy',
            evaluate_score=False
        )
        print(time() - t0)
        self._tprint("Evaluating x_test_bsm",)
        t0 = time() 
        self.log_r_hat_bsm, _, _ = self.forge.evaluate(
            theta0_filename='data/samples/theta_grid.npy',
            x='data/samples/x_test_bsm.npy',
            evaluate_score=False
        )
        print(time() - t0)
        self._tprint("Evaluating x_test_bsm_morph",)
        t0 = time() 
        self.log_r_hat_bsm_morph, _, _ = self.forge.evaluate(
            theta0_filename='data/samples/theta_grid.npy',
            x='data/samples/x_test_bsm_morph.npy',
            evaluate_score=False
        )
        self.has_evaluate = True
        print(time() - t0)

    def plot_results(self):
        if not self.has_evaluate:
            self._tprint("QUITTING: object has not yet evaluated any data!")
            return 1
        #bin_size = theta_each[1] - theta_each[0]
        #edges = np.linspace(theta_each[0] - bin_size/2, theta_each[-1] + bin_size/2, len(theta_each)+1)

        plt.figure(figsize=(6,5))
        #ax = plt.gca()

        expected_llr           = np.mean(self.log_r_hat,axis=1)
        expected_llr_bsm       = np.mean(self.log_r_hat_bsm,axis=1)
        expected_llr_bsm_morph = np.mean(self.log_r_hat_bsm_morph,axis=1)

        best_fit           = self.theta_grid[np.argmin(-2.*expected_llr)]
        best_fit_bsm       = self.theta_grid[np.argmin(-2.*expected_llr_bsm)]
        best_fit_bsm_morph = self.theta_grid[np.argmin(-2.*expected_llr_bsm_morph)]

        #cmin, cmax = np.min(-2*expected_llr), np.max(-2*expected_llr)

        print("CP even:", expected_llr)
        print("CP odd:", expected_llr_bsm)
        print("mixed:", expected_llr_bsm_morph)
        print(self.theta_each)
        print("best fit point for CP even at:", best_fit)
        print("best fit point for CP odd at:", best_fit_bsm)
        print("best fit point for mixed at:", best_fit_bsm_morph)

        plt.plot(self.theta_each, -2*expected_llr, label='CP even')
        plt.plot(self.theta_each, -2*expected_llr_bsm, label='CP odd')
        plt.plot(self.theta_each, -2*expected_llr_bsm_morph, label='mixed')

        #pcm = ax.pcolormesh(edges, edges, -2. * expected_llr.reshape((15,15)),
        #                    norm=matplotlib.colors.Normalize(vmin=cmin, vmax=cmax),
        #                    cmap='viridis_r')
        #cbar = fig.colorbar(pcm, ax=ax, extend='both')
        #
        #plt.scatter(best_fit[0], best_fit[1], s=80., color='black', marker='*')

        plt.xlabel(r'$\theta_0 = \cos\left(\alpha\right)$')
        #plt.ylabel(r'$\theta_1$')
        plt.ylabel(r'$\mathbb{E}_x [ -2\, \log \,\hat{r}(x | \theta, \theta_{SM}) ]$ (ALICE)')

        plt.legend()
        plt.tight_layout()
        plt.show()

class tth:
    def __init__(self, use_parton_level=True, n_samples=100000):
        self.use_parton_level = use_parton_level
        self.n_samples = n_samples

        self.sampling = sampling(use_parton_level=self.use_parton_level,
                                 n_samples=self.n_samples)
        self.training = training()
        self.validation = validation()

    def run(self):
        self.sampling.run()
        self.training.run()
        self.validation.run()