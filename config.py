from _config_section import ConfigSection

import os
REAL_PATH = os.path.dirname(os.path.realpath(__file__))

data = ConfigSection("data")
data.dir = "%s/%s" % (REAL_PATH, "data")

img = ConfigSection("img")
img.dir = "%s/%s" % (data.dir, "img")

hdf5 = ConfigSection("hdf5")
hdf5.dir = "%s/%s" % (data.dir, "hdf5")

shots = ConfigSection("shots")
shots.dir = "%s/%s" % (img.dir, "shots")

vae_shots = ConfigSection("vae_shots")
vae_shots.dir = "%s/%s" % (img.dir, "vae_shots")

plots = ConfigSection("plots")
plots.dir = "%s/%s" % (REAL_PATH, "plots")
