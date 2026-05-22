#!/usr/bin/env python3
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import keras
from keras import layers

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import datetime
from utils import *

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts-run-all-tensorflow"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = ARTIFACTS_DIR / "example_dataset.npz"
WEIGHTS_PATH = ARTIFACTS_DIR / "weights.keras"
MODEL_PATH = ARTIFACTS_DIR / "example_network.keras"
LOSS_PLOT_PATH = ARTIFACTS_DIR / "training_loss.png"
DECAY_PLOT_PATH = ARTIFACTS_DIR / "velocity_decay.png"
FIELDS_PLOT_DIR = ARTIFACTS_DIR / "velocity_fields"
FIELDS_PLOT_DIR.mkdir(parents=True, exist_ok=True)


def compute_rho_u(num_samples, rho_min=0.95, rho_max=1.05, u_abs_min=0.0, u_abs_max=0.01):

    rho = np.random.uniform(rho_min, rho_max, size=num_samples)
    u_abs = np.random.uniform(u_abs_min, u_abs_max, size=num_samples)
    theta = np.random.uniform(0, 2 * np.pi, size=num_samples)

    ux = u_abs * np.cos(theta)
    uy = u_abs * np.sin(theta)
    u = np.array([ux, uy]).transpose()

    return rho, u


def compute_f_rand(num_samples, sigma_min, sigma_max):

    Q = 9
    K0 = 1 / 9.0
    K1 = 1 / 6.0

    #########################################

    f_rand = np.zeros((num_samples, Q))

    #########################################

    if sigma_min == sigma_max:
        sigma = sigma_min * np.ones(num_samples)
    else:
        sigma = np.random.uniform(sigma_min, sigma_max, size=num_samples)

    #########################################

    for i in range(num_samples):
        f_rand[i, :] = np.random.normal(0, sigma[i], size=(1, Q))

        rho_hat = np.sum(f_rand[i, :])
        ux_hat = np.sum(f_rand[i, :] * c[:, 0])
        uy_hat = np.sum(f_rand[i, :] * c[:, 1])

        f_rand[i, :] = f_rand[i, :] - K0 * rho_hat - K1 * ux_hat * c[:, 0] - K1 * uy_hat * c[:, 1]

    return f_rand


def compute_f_pre_f_post(f_eq, f_neq, tau_min=1, tau_max=1):

    tau = np.random.uniform(tau_min, tau_max, size=f_eq.shape[0])
    f_pre = f_eq + f_neq

    f_post = f_pre + 1 / tau[:, None] * (f_eq - f_pre)

    return tau, f_pre, f_post


def delete_negative_samples(n_samples, f_eq, f_pre, f_post):

    i_neg_f_eq = np.where(np.sum(f_eq < 0, axis=1) > 0)[0]
    i_neg_f_pre = np.where(np.sum(f_pre < 0, axis=1) > 0)[0]
    i_neg_f_post = np.where(np.sum(f_post < 0, axis=1) > 0)[0]

    i_neg_f = np.concatenate((i_neg_f_pre, i_neg_f_post, i_neg_f_eq))

    f_eq = np.delete(np.copy(f_eq), i_neg_f, 0)
    f_pre = np.delete(np.copy(f_pre), i_neg_f, 0)
    f_post = np.delete(np.copy(f_post), i_neg_f, 0)

    return f_eq, f_pre, f_post


def load_data(fname):

    data = np.load(fname, allow_pickle=True)

    feq = data["f_eq"]
    fpre = data["f_pre"]
    fpost = data["f_post"]

    return feq, fpre, fpost


def sequential_model(Q=9, n_hidden_layers=2, n_per_layer=50, activation="relu", ll_activation="linear", bias=False):

    model = Sequential(
        [
            keras.Input(shape=(Q,)),
            Dense(n_per_layer, activation=activation, use_bias=bias, kernel_initializer="he_uniform"),
        ]
    )

    for jj in range(n_hidden_layers):
        model.add(Dense(n_per_layer, activation=activation, use_bias=bias, kernel_initializer="he_uniform"))

    model.add(Dense(Q, activation=ll_activation, use_bias=bias, kernel_initializer="he_uniform"))

    return model


def create_model(
    loss="mape",
    optimizer="adam",
    Q=9,
    n_hidden_layers=2,
    n_per_layer=50,
    activation="relu",
    ll_activation="linear",
    bias=False,
):

    the_input = keras.Input(shape=(Q,))

    seq_model = sequential_model(Q, n_hidden_layers, n_per_layer, ll_activation, ll_activation, bias)

    input_lst = D4Symmetry()(the_input)

    output_lst = [seq_model(x) for k, x in enumerate(input_lst)]

    output_lst = [AlgReconstruction()(input_lst[k], x) for k, x in enumerate(output_lst)]

    output_lst = D4AntiSymmetry()(output_lst)

    the_output = layers.Average()(output_lst)

    model = keras.Model(inputs=the_input, outputs=the_output)

    model.compile(loss=loss, optimizer=optimizer)

    return model


def data_collector(dumpfile, t, ux, uy, rho):
    it = t // dumpit
    idx0 = it * (nx * ny)
    idx1 = (it + 1) * (nx * ny)
    dumpfile[idx0:idx1, 0] = t
    dumpfile[idx0:idx1, 1] = rho.reshape(nx * ny)
    dumpfile[idx0:idx1, 2] = ux.reshape(nx * ny)
    dumpfile[idx0:idx1, 3] = uy.reshape(nx * ny)


def sol(t, L, F0, nu):
    return F0 * np.exp(-2 * nu * t / (L / (2 * np.pi)) ** 2)


#########################################################
#
# Create Training Data
#
#########################################################

#####################################
# settings

n_samples = 100_000

u_abs_min = 1e-15
u_abs_max = 0.01
sigma_min = 1e-15
sigma_max = 5e-4

#####################################
# lattice velocities and weights
Q = 9
c, w, cs2, compute_feq = LB_stencil()

#####################################

fPreLst = np.empty((n_samples, Q))
fPostLst = np.empty((n_samples, Q))
fEqLst = np.empty((n_samples, Q))

#####################################

idx = 0

# loop until we get n_samples without negative populations
while idx < n_samples:

    # get random values for macroscopic quantities
    rho, u = compute_rho_u(n_samples)

    rho = rho[:, np.newaxis]
    ux = u[:, 0][:, np.newaxis]
    uy = u[:, 1][:, np.newaxis]

    # compute the equilibrium distribution
    f_eq = np.zeros((n_samples, 1, Q))
    f_eq = compute_feq(f_eq, rho, ux, uy, c, w)[:, 0, :]

    # compute a random non equilibrium part
    f_neq = compute_f_rand(n_samples, sigma_min, sigma_max)

    # apply BGK to f_pre = f_eq + f_neq
    tau, f_pre, f_post = compute_f_pre_f_post(f_eq, f_neq)

    # remove negative elements
    f_eq, f_pre, f_post = delete_negative_samples(n_samples, f_eq, f_pre, f_post)

    # accumulate
    non_negatives = f_pre.shape[0]

    idx1 = min(idx + non_negatives, n_samples)
    to_be_added = min(n_samples - idx, non_negatives)

    fPreLst[idx:idx1] = f_pre[:to_be_added]
    fPostLst[idx:idx1] = f_post[:to_be_added]
    fEqLst[idx:idx1] = f_eq[:to_be_added]

    idx = idx + non_negatives


# store data on file

np.savez(DATASET_PATH, f_pre=fPreLst, f_post=fPostLst, f_eq=fEqLst)


#########################################################
#
# Training
#
#########################################################
# set precision (default is 'float32')
K.set_floatx("float64")

# read training dataset
feq, fpre, fpost = load_data(DATASET_PATH)

# normalize data on density
feq = feq / np.sum(feq, axis=1)[:, np.newaxis]
fpre = fpre / np.sum(fpre, axis=1)[:, np.newaxis]
fpost = fpost / np.sum(fpost, axis=1)[:, np.newaxis]

# split train and test set
fpre_train, fpre_test, fpost_train, fpost_test = train_test_split(fpre, fpost, test_size=0.3, shuffle=True)

batch_size = 32
n_epochs = 200
patience = 50
verbose = 1

model = create_model(loss=rmsre, ll_activation="softmax")

# EarlyStopping
es_callback = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

# Save best model during training
ck_callback = ModelCheckpoint(filepath=str(WEIGHTS_PATH), monitor="val_loss", save_best_only=True)

keras_callbacks = [es_callback, ck_callback]

## training the model
hist = model.fit(
    fpre_train,
    fpost_train,
    epochs=n_epochs,
    verbose=verbose,
    callbacks=keras_callbacks,
    validation_data=(fpre_test, fpost_test),
    batch_size=batch_size,
)

model.load_weights(str(WEIGHTS_PATH))
model.save(str(MODEL_PATH))

model.evaluate(fpre_test, fpost_test)

plt.figure()
plt.semilogy(hist.history["loss"], lw=3, label="Training")
plt.semilogy(hist.history["val_loss"], lw=3, label="Validation")

plt.legend(loc="best", frameon=False)

plt.savefig(LOSS_PLOT_PATH, dpi=120, bbox_inches="tight")
plt.close()


#########################################################
#
# Reconstruction of the collision operator / Simulation
#
#########################################################
# set precision (default is 'float32')
K.set_floatx("float64")

##########################################################
# Import trained model from file

model = keras.models.load_model(str(MODEL_PATH), custom_objects={"rmsre": rmsre})
model.summary()

###########################################################
# Simulation Parameters

nx = 32  # grid size along x
ny = 32  # grid size along y
niter = 1000  # total number of steps
dumpit = 100  # collect data every dumpit iterations
tau = 1.0  # relaxation time
u0 = 0.01  # initial velocity amplitude

verbose = 0


###########################################################
# Collect stats
ndumps = int(niter // dumpit)
dumpfile = np.zeros((ndumps * nx * ny, 4))
###########################################################


##########################################################
# Set Initial conditions

a = b = 1.0

ix, iy = np.meshgrid(range(nx), range(ny), indexing="ij")

x = 2.0 * np.pi * (ix / nx)
y = 2.0 * np.pi * (iy / ny)

ux = 1.0 * u0 * np.sin(a * x) * np.cos(b * y)
uy = -1.0 * u0 * np.cos(a * x) * np.sin(b * y)

rho = np.ones((nx, ny))

###########################################################
# Lattice velocities and weights
Q = 9
c, w, cs2, compute_feq = LB_stencil()

###########################################################
# Lattice
feq = np.zeros((nx, ny, Q))
feq = compute_feq(feq, rho, ux, uy, c, w)

f1 = np.copy(feq)
f2 = np.copy(feq)


###########################################################

data_collector(dumpfile, 0, ux, uy, rho)

###########################################################

m_initial = np.sum(f1.flatten())

###########################################################
# Loop on time steps
for t in range(1, niter):

    # streaming
    for ip in range(Q):
        f1[:, :, ip] = np.roll(np.roll(f2[:, :, ip], c[ip, 0], axis=0), c[ip, 1], axis=1)

    # Calculate density
    rho = np.sum(f1, axis=2)

    # Calculate velocity
    ux = (1.0 / rho) * np.einsum("ijk,k", f1, c[:, 0])
    uy = (1.0 / rho) * np.einsum("ijk,k", f1, c[:, 1])

    #########################################
    # ML collision step
    #########################################

    # Normalize input data
    fpre = f1.reshape((nx * ny, Q))
    norm = np.sum(fpre, axis=1)[:, np.newaxis]
    fpre = fpre / norm

    # Make prediction
    f2 = model.predict(fpre, verbose=verbose)

    # Rescale output
    f2 = norm * f2
    f2 = f2.reshape((nx, ny, Q))

    #########################################

    # Collect data
    if (t % dumpit) == 0:
        data_collector(dumpfile, t, ux, uy, rho)

m_final = np.sum(f2.flatten())


print("Sim ended. Mass err:", np.abs(m_initial - m_final) / m_initial)

w = 3.46 * 3
h = 2.14 * 3


fig = plt.figure(figsize=(w, h))
ax = fig.add_subplot(111)

tLst = np.arange(0, niter, dumpit)

for i, t in enumerate(tLst):

    ux = dumpfile[dumpfile[:, 0] == t, 2]
    uy = dumpfile[dumpfile[:, 0] == t, 3]

    Ft = np.average((ux**2 + uy**2) ** 0.5)

    if i == 0:
        F0 = Ft
        ax.semilogy(t, Ft, "ob", label="lbm")
    else:
        ax.semilogy(t, Ft, "ob")

nu = (tau - 0.5) * (cs2)

ax.semilogy(tLst, sol(tLst, nx, F0, nu), linewidth=2.0, linestyle="--", color="r", label="analytic")

###################################################################

ax.set_xlabel(r"$t~\rm{[L.U.]}$", fontsize=16)
ax.set_ylabel(r"$\langle |u| \rangle$", fontsize=16, rotation=90, labelpad=0)

ax.legend(loc="best", frameon=False, prop={"size": 16})

ax.tick_params(which="both", direction="in", top="on", right="on", labelsize=14)

fig.savefig(DECAY_PLOT_PATH, dpi=120, bbox_inches="tight")
plt.close(fig)


w = 3.46 * 3
h = 2.14 * 3

X, Y = np.meshgrid(np.arange(0, nx), np.arange(0, ny))

tLst = np.arange(0, niter, dumpit)

for i, t in enumerate(tLst):

    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)

    ux = dumpfile[dumpfile[:, 0] == t, 2].reshape((nx, ny))
    uy = dumpfile[dumpfile[:, 0] == t, 3].reshape((nx, ny))

    u = (ux**2 + uy**2) ** 0.5

    vmin = 0
    vmax = 1e-2

    im = ax.imshow(u)  # , vmax=vmax, vmin=vmin)

    ax.streamplot(X, Y, ux, uy, density=0.5, color="w")

    fig.colorbar(im, ax=ax, orientation="vertical", pad=0, shrink=0.69)

    ax.set_title(f"Iteration {t}", size=16)

    fig.savefig(FIELDS_PLOT_DIR / f"velocity_field_t{int(t):05d}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
