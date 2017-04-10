from _config_section import ConfigSection

import os
REAL_PATH = os.path.dirname(os.path.realpath(__file__))

data = ConfigSection("data")
data.dir = "%s/%s/" % (REAL_PATH, "data")

plots = ConfigSection("plots")
plots.dir = "%s/%s/" % (REAL_PATH, "plots")
