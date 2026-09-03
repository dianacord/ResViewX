"""Local resolution — skeleton with fake prototype function.

Parameter names mirror the real MonoRes protocol (xmipp) so the UI is ready
for the real algorithm to drop in later:
  min_res      — high-resolution limit (Å), default 0/1
  max_res      — low-resolution limit (Å)
  step         — resolution sampling interval (Å), default 0.5
  significance — hypothesis test stringency, default 0.95
"""

import numpy as np


def fake_monores(map1, map2=None, mask=None,
                 min_res=1.0, max_res=15.0, step=0.5, significance=0.95):
    """Fake local resolution output for UI prototyping.

    Returns a random array the same shape as map1 with values in [2, 12] Å.
    Replace this with the real MonoRes / local resolution algorithm later.
    Voxels outside the mask (mask == 0) are set to NaN so they are ignored
    downstream, mimicking how real local-resolution only reports inside the mask.
    """
    fake_output = 2 + 10 * np.random.rand(*map1.shape)
    fake_output = fake_output.astype(np.float32)
    if mask is not None:
        fake_output[mask <= 0] = np.nan
    return fake_output
