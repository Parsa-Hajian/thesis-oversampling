"""
wrapper.py -- Unified oversampler factory.

Returns a callable() -> oversampler for use with LeakageSafeCV.
All oversampler instances are created fresh per call (stateless factories).
"""

import sys
import os

_CO_ROOT = os.path.expanduser("~/Desktop/circular-oversampling")
if _CO_ROOT not in sys.path:
    sys.path.insert(0, _CO_ROOT)

from imblearn.over_sampling import (
    SMOTE,
    BorderlineSMOTE,
    ADASYN,
    KMeansSMOTE,
    RandomOverSampler,
)


def _try_import_gvm():
    try:
        from src.core.gravity_vonmises import GravityVonMises
        return GravityVonMises
    except ImportError:
        return None


def _try_import_lre():
    try:
        from src.core.local_regions import LocalRegionsOversampler
        return LocalRegionsOversampler
    except ImportError:
        return None


def _try_import_ls():
    try:
        from src.core.layered_segmental import LayeredSegmentalOversampler
        return LayeredSegmentalOversampler
    except ImportError:
        return None


def get_oversampler_factory(name: str, random_state: int = 42):
    """
    Return a zero-argument callable that instantiates the named oversampler.

    Supported names
    ---------------
    none          : no-op (returns X, y unchanged)
    ros           : Random Over-Sampling
    smote         : SMOTE
    bsmote        : Borderline-SMOTE
    adasyn        : ADASYN
    kmsmote       : K-Means SMOTE
    gvmco         : GravityVonMises (from circular-oversampling)
    lreco         : LocalRegionsOversampler
    lsco          : LayeredSegmentalOversampler
    """
    name = name.lower()

    if name == "none":
        class NoOp:
            def fit_resample(self, X, y):
                return X, y
        return lambda: NoOp()

    if name == "ros":
        return lambda: RandomOverSampler(random_state=random_state)

    if name == "smote":
        return lambda: SMOTE(random_state=random_state, k_neighbors=5)

    if name == "bsmote":
        return lambda: BorderlineSMOTE(random_state=random_state, k_neighbors=5)

    if name == "adasyn":
        return lambda: ADASYN(random_state=random_state)

    if name == "kmsmote":
        return lambda: KMeansSMOTE(random_state=random_state)

    if name == "gvmco":
        GVM = _try_import_gvm()
        if GVM is None:
            raise ImportError("GravityVonMises not available from circular-oversampling")
        return lambda: GVM(random_state=random_state)

    if name == "lreco":
        LRE = _try_import_lre()
        if LRE is None:
            raise ImportError("LocalRegionsOversampler not available")
        return lambda: LRE(random_state=random_state)

    if name == "lsco":
        LS = _try_import_ls()
        if LS is None:
            raise ImportError("LayeredSegmentalOversampler not available")
        return lambda: LS(random_state=random_state)

    raise ValueError(f"Unknown oversampler: {name!r}")


ALL_OVERSAMPLERS = ["none", "ros", "smote", "bsmote", "adasyn", "kmsmote",
                    "gvmco", "lreco", "lsco"]
