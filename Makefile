NULL =

APPLESEED_CLI ?= appleseed.cli

all:

results/%.pickle: main/%.py main/ma.py
	$<

results/optics.pickle: optics/image.png

figures/out/%.png: figures/%.py figures/common.py
	$<

figures/out/%_report.txt: figures/%.py figures/common.py
	$<

figures/out/transport_maps.png: \
		results/const_to_const.pickle \
		results/const_to_gaussians.pickle \
		results/ring_to_const.pickle \
		results/ring_to_gaussians.pickle \
		results/angle_to_const.pickle \
		results/angle_to_gaussians.pickle \
		results/three_to_const.pickle \
		results/three_to_gaussians.pickle \
		results/rand_to_const.pickle \
		results/rand_to_gaussians.pickle \
		$(NULL)

figures/out/errors.png: \
		results/errors_quartic_sb2.pickle \
		results/errors_c1_sb2.pickle \
		$(NULL)

figures/out/errors_dir.png: \
		results/errors_quartic_sb2_dir.pickle \
		results/errors_quartic_sb6_dir.pickle \
		results/errors_quartic_sb10_dir.pickle \
		results/errors_c1_sb2_dir.pickle \
		results/errors_c1_sb6_dir.pickle \
		results/errors_c1_sb10_dir.pickle \
		results/errors_singular_sb2_dir.pickle \
		results/errors_singular_sb6_dir.pickle \
		results/errors_singular_sb10_dir.pickle \
		$(NULL)

figures/out/abs.png: \
		results/abs_sb2.pickle \
		results/abs_sb6.pickle \
		results/abs_sb10.pickle \
		$(NULL)

figures/out/newton_steps_comp_malbr_report.txt: \
		results/convex_dir.pickle \
		results/convex_dir_lbr.pickle \
		results/nonconvex_dir.pickle \
		results/nonconvex_dir_lbr.pickle \
		$(NULL)

figures/out/optics_target.png: results/optics.pickle

figures/out/optics_curvature.png: results/optics.pickle

optics/meshes/%.binarymesh: optics/gen%.py optics/genmesh.py
	$<

optics/meshes/lens.binarymesh: results/optics.pickle

figures/out/optics_simulation.png: \
		optics/simulate.sh \
		optics/simulation.appleseed \
		optics/meshes/circle.binarymesh \
		optics/meshes/square.binarymesh \
		optics/meshes/lens.binarymesh \
		$(NULL)
	$< $(APPLESEED_CLI)

all-but-optics: \
		figures/out/transport_maps.png \
		figures/out/errors.png \
		figures/out/errors_dir.png \
		figures/out/abs.png \
		figures/out/newton_steps_comp_malbr_report.txt \
		$(NULL)
.PHONY: all-but-optics

optics: \
		figures/out/optics_target.png \
		figures/out/optics_curvature.png \
		figures/out/optics_simulation.png \
		$(NULL)
.PHONY: optics

all: all-but-optics optics
.PHONY: all

clean:
	-rm -rf results/*.pickle
	-rm -rf figures/out/*.png
	-rm -rf figures/out/*.txt
	-rm -rf optics/meshes/*.binarymesh
.PHONY: clean
