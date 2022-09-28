#!/usr/bin/env python3

import os

import common

results = common.LoadResults("convex_dir.pickle")

niter_convex_dir = results["niter"]

results = common.LoadResults("convex_dir_lbr.pickle")

dampingSteps_convex_dir_lbr = results["dampingSteps"]
niter_convex_dir_lbr = len(dampingSteps_convex_dir_lbr)

results = common.LoadResults("nonconvex_dir.pickle")

niter_nonconvex_dir = results["niter"]

results = common.LoadResults("nonconvex_dir_lbr.pickle")

dampingSteps_nonconvex_dir_lbr = results["dampingSteps"]
niter_nonconvex_dir_lbr = len(dampingSteps_nonconvex_dir_lbr)

with open(
    os.path.join(os.path.dirname(__file__), "out/newton_steps_comp_malbr_report.txt"),
    "w",
) as file:
    print("*** convex_dir ***", file=file)
    print("niter:", niter_convex_dir, file=file)

    print("*** convex_dir_lbr ***", file=file)
    print("niter:", niter_convex_dir_lbr, file=file)
    print("damping steps:", dampingSteps_convex_dir_lbr, file=file)

    print("*** nonconvex_dir ***", file=file)
    print("niter:", niter_nonconvex_dir, file=file)

    print("*** nonconvex_dir_lbr ***", file=file)
    print("niter:", niter_nonconvex_dir_lbr, file=file)
    print("damping steps:", dampingSteps_nonconvex_dir_lbr, file=file)
