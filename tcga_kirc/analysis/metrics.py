"""Association metrics between patient-level pseudotime and clinical endpoints."""

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy.stats import kendalltau


def tau_b(pseudotime: pd.Series, target: pd.Series) -> dict:
    """Kendall's tau-b between pseudotime and an ordinal or continuous target.

    scipy's default variant is 'b', which is the tie-corrected form needed for
    the IDH ordinal (many ties across only three levels).
    """
    mask = pseudotime.notna() & target.notna()
    if mask.sum() < 3:
        return {"tau": np.nan, "p": np.nan, "n": int(mask.sum())}

    tau, p = kendalltau(pseudotime[mask], target[mask], variant="b")
    return {"tau": tau, "p": p, "n": int(mask.sum())}


def c_index(pseudotime: pd.Series, time: pd.Series, event: pd.Series) -> dict:
    """Harrell's C-index treating pseudotime as a risk score.

    lifelines expects predicted_scores oriented like survival time (higher =
    longer survival), so the risk score is negated. C > 0.5 therefore means
    higher pseudotime predicts shorter survival, i.e. pseudotime tracks
    aggressiveness in the expected direction.
    """
    mask = pseudotime.notna() & time.notna() & event.notna()
    if mask.sum() < 3 or event[mask].sum() == 0:
        return {"c_index": np.nan, "n": int(mask.sum()), "n_events": 0}

    c = concordance_index(time[mask], -pseudotime[mask], event[mask])
    return {"c_index": c, "n": int(mask.sum()), "n_events": int(event[mask].sum())}
