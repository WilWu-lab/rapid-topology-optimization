Fig. 8 data export
==================

Grid: 301 x 301 pixels, 1 mm step, x = 100..400 mm, y = 100..400 mm.
Row index = y, column index = x. Values outside the L-shaped plate are NaN.

Cases:
  optimized          optimized layout, per-path beta (Fig. 8a)
  baseline_beta101   valid staggered 3x2 baseline, beta = 1.01 (Fig. 8b, row 1)
  baseline_beta102   valid staggered 3x2 baseline, beta = 1.02 (Fig. 8b, row 2)
  baseline_beta103   valid staggered 3x2 baseline, beta = 1.03 (Fig. 8b, row 3)

Files:
  x_mm.csv, y_mm.csv              axis vectors
  P_<case>.csv                    path count N (0,1,2,...), NaN outside plate
  mask_<case>_blind.csv           1 where N=0, NaN outside plate
  mask_<case>_weak.csv            1 where N=1, NaN outside plate
  mask_<case>_effective.csv       1 where N>=2, NaN outside plate
  long_<case>.csv                 long format with x,y,N and the three masks
  summary.csv                     coverage rates and target-2 RMSE
  nodes.csv                       transducer/sensor coordinates in mm

Colors in the published Fig. 8: blind = red, weak = orange/yellow, effective = green.
