"""
pka_predictor.py
================

A modular, class-based framework for estimating the pKa values of small
molecules from SMILES strings.

The pipeline is split into three responsibilities:

    MoleculeProcessor  ->  clean & canonicalise structures (salts, stereo, etc.)
    FeatureExtractor   ->  ECFP4 fingerprints + physicochemical descriptors
    pKaPredictor       ->  pluggable ML model interface (+ rule-based fallback)

CHEMICAL CONTEXT
----------------
pKa is the negative log of the acid-dissociation constant. In medicinal
chemistry we care about it because the *ionisation state at physiological
pH (7.4)* drives solubility, membrane permeability, plasma-protein binding
and target engagement.

By convention pKa is always reported for the *conjugate-acid <-> conjugate-base*
equilibrium:

    Acid   HA  <->  A(-) + H(+)      low  pKa  => stronger acid
    Base   BH(+) <->  B  + H(+)      high pKa  => stronger base

A single molecule can have BOTH acidic and basic centres (it is
*polyprotic* / *amphoteric*), so a real tool must never collapse a molecule
to a single number. This framework enumerates every ionisable centre and
predicts each one independently.

Author: (template) Senior Cheminformatics Engineer
License: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

# Silence RDKit's very chatty C++ logger; we do our own logging.
RDLogger.DisableLog("rdApp.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("pka_predictor")


# --------------------------------------------------------------------------- #
#  Ionisable-centre SMARTS definitions
# --------------------------------------------------------------------------- #
#  These curated SMARTS patterns let us *locate* acidic / basic atoms. This is
#  what makes polyprotic handling possible: instead of one number per molecule
#  we tag each matching atom and predict a pKa for it individually.
#
#  Rationale for the choices below:
#    * Carboxylic acids / sulfonic acids / phenols / tetrazoles / imides are the
#      dominant *acidic* motifs in drug-like space.
#    * Aliphatic amines, amidines, guanidines and basic aromatic N are the
#      dominant *basic* motifs.
#  The list is intentionally editable — extend it for your chemotype coverage.
# --------------------------------------------------------------------------- #
ACIDIC_SMARTS = {
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "sulfonic_acid": "[SX4](=O)(=O)[OX2H1]",
    "phosphonic_acid": "[PX4](=O)([OX2H1])",
    "phenol": "[c][OX2H1]",
    "tetrazole": "c1nnn[nH]1",
    "sulfonamide_acidic": "[SX4](=O)(=O)[NX3H1]",
    "imide": "[NX3H1](C=O)C=O",
    "thiol": "[#6][SX2H1]",
}

BASIC_SMARTS = {
    "aliphatic_amine": "[NX3;H2,H1,H0;!$(NC=O);!$(N=*);!$([N+]);!$(Nc)]",
    "amidine": "[NX3][CX3]=[NX2]",
    "guanidine": "[NX3][CX3](=[NX2])[NX3]",
    "aromatic_n_basic": "[nX2;!$(n[#6]=O)]",  # e.g. pyridine-type N
}


@dataclass
class IonisableCentre:
    """A single acid/base centre detected on a molecule."""

    atom_idx: int
    group_name: str
    acid_or_base: str  # "acidic" | "basic"


@dataclass
class ProcessedMolecule:
    """Container for a cleaned molecule and its provenance."""

    mol_id: str
    input_smiles: str
    canonical_smiles: Optional[str] = None
    mol: Optional[Chem.Mol] = None
    error: Optional[str] = None
    centres: List[IonisableCentre] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.mol is not None and self.error is None


# --------------------------------------------------------------------------- #
#  1. MoleculeProcessor
# --------------------------------------------------------------------------- #
class MoleculeProcessor:
    """Clean, canonicalise and standardise incoming SMILES.

    Responsibilities
    ----------------
    * Reject syntactically invalid SMILES.
    * Strip counter-ions / salts by keeping the largest organic fragment
      (a molecule of interest is almost never the sodium or chloride).
    * Normalise functional groups (e.g. nitro, tautomer-adjacent forms) so
      that identical chemistry maps to identical descriptors.
    * Preserve stereochemistry in the canonical SMILES — stereo can matter for
      3D-dependent properties, though classical pKa is largely 2D-driven.
    """

    def __init__(self, keep_stereochemistry: bool = True) -> None:
        self.keep_stereochemistry = keep_stereochemistry
        # RDKit standardization helpers, instantiated once for speed.
        self._largest_fragment = rdMolStandardize.LargestFragmentChooser()
        self._normalizer = rdMolStandardize.Normalizer()
        self._uncharger = rdMolStandardize.Uncharger()

    def process(self, smiles: str, mol_id: str) -> ProcessedMolecule:
        """Return a :class:`ProcessedMolecule` for one SMILES string."""
        record = ProcessedMolecule(mol_id=mol_id, input_smiles=smiles)

        if not isinstance(smiles, str) or not smiles.strip():
            record.error = "empty_or_non_string_smiles"
            return record

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            record.error = "invalid_smiles"
            return record

        try:
            # 1. keep the largest fragment (drops salts / solvents)
            mol = self._largest_fragment.choose(mol)
            # 2. normalise functional-group representations
            mol = self._normalizer.normalize(mol)
            # 3. neutralise cleanly-chargeable atoms so the *neutral* form is
            #    what we featurise; ionisation is what we are trying to predict.
            mol = self._uncharger.uncharge(mol)
            Chem.SanitizeMol(mol)
        except Exception as exc:  # noqa: BLE001 - report any RDKit failure
            record.error = f"standardization_failed:{exc}"
            return record

        record.mol = mol
        record.canonical_smiles = Chem.MolToSmiles(
            mol, isomericSmiles=self.keep_stereochemistry
        )
        record.centres = self.find_ionisable_centres(mol)
        return record

    @staticmethod
    def find_ionisable_centres(mol: Chem.Mol) -> List[IonisableCentre]:
        """Locate every acidic and basic atom via SMARTS matching.

        This is the mechanism that enables *polyprotic* handling: we return one
        entry per matched centre rather than a single molecule-level label.
        """
        centres: List[IonisableCentre] = []

        for name, smarts in ACIDIC_SMARTS.items():
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            for match in mol.GetSubstructMatches(patt):
                centres.append(
                    IonisableCentre(match[0], name, "acidic")
                )

        for name, smarts in BASIC_SMARTS.items():
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            for match in mol.GetSubstructMatches(patt):
                centres.append(
                    IonisableCentre(match[0], name, "basic")
                )

        # De-duplicate on (atom_idx, acid_or_base) — a single atom may be hit
        # by overlapping patterns (e.g. guanidine vs amidine).
        seen = set()
        unique: List[IonisableCentre] = []
        for c in centres:
            key = (c.atom_idx, c.acid_or_base)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique


# --------------------------------------------------------------------------- #
#  2. FeatureExtractor
# --------------------------------------------------------------------------- #
class FeatureExtractor:
    """Turn a cleaned molecule into a numeric feature vector.

    Two complementary feature families are used, which is standard practice for
    QSPR/pKa modelling:

    ECFP4 (Morgan, radius=2) fingerprints
        Encode the *local chemical environment* around each atom. Because pKa is
        governed by the electronic environment immediately surrounding the
        ionisable atom (inductive / resonance effects of neighbours), circular
        fingerprints are a strong, information-dense signal.

    Physicochemical descriptors
        Cheap, interpretable global properties that correlate with ionisation:

        * MolWt            - crude proxy for size / complexity.
        * MolLogP          - lipophilicity; anti-correlated with polar,
                              ionisable surface area.
        * TPSA             - topological polar surface area; ionisable groups are
                              polar, so TPSA tracks H-bonding / charge capacity.
        * NumHDonors       - donors (O-H, N-H) are the atoms that *lose* protons
                              -> directly relevant to acidic pKa.
        * NumHAcceptors    - acceptors (lone-pair N/O) are the atoms that *gain*
                              protons -> directly relevant to basic pKa.
        * NumRotatableBonds- flexibility; weak/indirect but cheap.
        * NumAromaticRings - aromatic delocalisation strongly stabilises
                              conjugate bases/acids (e.g. phenol vs alcohol).
        * FractionCSP3     - saturation; distinguishes aliphatic vs aromatic
                              amine basicity, which differ by several pKa units.
    """

    #: order matters — this is the descriptor column order used everywhere.
    DESCRIPTOR_FUNCS = {
        "MolWt": Descriptors.MolWt,
        "MolLogP": Crippen.MolLogP,
        "TPSA": rdMolDescriptors.CalcTPSA,
        "NumHDonors": rdMolDescriptors.CalcNumHBD,
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA,
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    }

    def __init__(self, fp_radius: int = 2, fp_n_bits: int = 2048) -> None:
        # radius=2 => ECFP4 (diameter 4). 2048 bits is the common default.
        self.fp_radius = fp_radius
        self.fp_n_bits = fp_n_bits
        self._fp_gen = AllChem.GetMorganGenerator(
            radius=fp_radius, fpSize=fp_n_bits
        )

    def morgan_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """Return the ECFP4 bit vector as a float numpy array."""
        fp = self._fp_gen.GetFingerprint(mol)
        arr = np.zeros((self.fp_n_bits,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """Return the physicochemical descriptor vector."""
        return np.array(
            [func(mol) for func in self.DESCRIPTOR_FUNCS.values()],
            dtype=np.float32,
        )

    def descriptor_names(self) -> List[str]:
        return list(self.DESCRIPTOR_FUNCS.keys())

    def extract(self, mol: Chem.Mol) -> np.ndarray:
        """Concatenate [descriptors | fingerprint] into one feature vector."""
        return np.concatenate(
            [self.descriptors(mol), self.morgan_fingerprint(mol)]
        )

    def feature_names(self) -> List[str]:
        return self.descriptor_names() + [
            f"ecfp4_{i}" for i in range(self.fp_n_bits)
        ]


# --------------------------------------------------------------------------- #
#  3. pKaPredictor
# --------------------------------------------------------------------------- #
class pKaPredictor:
    """Pluggable prediction layer.

    Design
    ------
    The heavy lifting (a *trained* regressor) is intentionally injected rather
    than hard-coded, so you can drop in scikit-learn, XGBoost, LightGBM or an
    ONNX runtime without touching the pipeline. We keep two independent models
    because acidic and basic pKa live on different physical scales and are best
    learned separately:

        acidic_model : predicts pKa of an acidic centre
        basic_model  : predicts pKa (of the conjugate acid) of a basic centre

    Any object exposing ``.predict(X)`` works. If the model is a *tree
    ensemble* exposing ``.estimators_`` (RandomForest) we derive a confidence
    metric from the spread of individual tree predictions — a cheap, honest
    epistemic-uncertainty proxy. If no model is supplied, a transparent
    rule-based estimator returns literature-average pKa per functional group so
    the pipeline is runnable out-of-the-box.
    """

    # Coarse literature-average pKa values, ONLY used by the fallback estimator.
    _FALLBACK_ACIDIC = {
        "carboxylic_acid": 4.5,
        "sulfonic_acid": -1.0,
        "phosphonic_acid": 2.0,
        "phenol": 10.0,
        "tetrazole": 4.9,
        "sulfonamide_acidic": 10.0,
        "imide": 9.5,
        "thiol": 8.5,
    }
    _FALLBACK_BASIC = {
        "aliphatic_amine": 10.5,
        "amidine": 12.4,
        "guanidine": 13.6,
        "aromatic_n_basic": 5.2,
    }

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        acidic_model=None,
        basic_model=None,
    ) -> None:
        self.features = feature_extractor
        self.acidic_model = acidic_model
        self.basic_model = basic_model

    # ---- model persistence ------------------------------------------------ #
    @classmethod
    def load(
        cls,
        feature_extractor: FeatureExtractor,
        acidic_model_path: Optional[str] = None,
        basic_model_path: Optional[str] = None,
    ) -> "pKaPredictor":
        """Load pickled models from disk (joblib format expected)."""
        import joblib

        acidic = joblib.load(acidic_model_path) if acidic_model_path else None
        basic = joblib.load(basic_model_path) if basic_model_path else None
        return cls(feature_extractor, acidic, basic)

    # ---- prediction helpers ---------------------------------------------- #
    def _predict_one(self, model, features: np.ndarray):
        """Return (value, confidence) for a single centre.

        Confidence is in [0, 1]; higher is better. For tree ensembles we map
        the inter-tree standard deviation to a confidence via a soft transform.
        """
        X = features.reshape(1, -1)
        value = float(model.predict(X)[0])

        if hasattr(model, "estimators_"):
            preds = np.array(
                [est.predict(X)[0] for est in model.estimators_]
            )
            std = float(preds.std())
            # 1.5 pKa-unit spread -> ~0.5 confidence; tune to your validation.
            confidence = float(np.exp(-std / 1.5))
        else:
            confidence = float("nan")  # unknown for opaque models
        return value, confidence

    def predict_molecule(self, record: ProcessedMolecule) -> List[dict]:
        """Predict a pKa for every ionisable centre of one molecule.

        Returns one row-dict per centre (polyprotic-aware). Molecules with no
        detected ionisable centre yield a single 'neutral' row.
        """
        if not record.is_valid:
            return [self._row(record, None, np.nan, np.nan, note=record.error)]

        if not record.centres:
            return [self._row(record, None, np.nan, np.nan, note="no_ionisable_centre")]

        features = self.features.extract(record.mol)
        rows: List[dict] = []
        for centre in record.centres:
            if centre.acid_or_base == "acidic":
                model = self.acidic_model
                fallback = self._FALLBACK_ACIDIC.get(centre.group_name, np.nan)
            else:
                model = self.basic_model
                fallback = self._FALLBACK_BASIC.get(centre.group_name, np.nan)

            if model is not None:
                value, confidence = self._predict_one(model, features)
            else:
                value, confidence = fallback, 0.25  # low, honest confidence

            rows.append(self._row(record, centre, value, confidence))
        return rows

    @staticmethod
    def _row(record, centre, value, confidence, note: str = "") -> dict:
        return {
            "mol_id": record.mol_id,
            "input_smiles": record.input_smiles,
            "canonical_smiles": record.canonical_smiles,
            "centre_atom_idx": centre.atom_idx if centre else np.nan,
            "functional_group": centre.group_name if centre else np.nan,
            "pka_type": centre.acid_or_base if centre else np.nan,
            "predicted_pka": round(value, 2) if value == value else np.nan,
            "confidence": round(confidence, 3) if confidence == confidence else np.nan,
            "note": note,
        }


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #
class PkaPipeline:
    """End-to-end: SMILES in -> tidy DataFrame out."""

    def __init__(
        self,
        processor: Optional[MoleculeProcessor] = None,
        extractor: Optional[FeatureExtractor] = None,
        predictor: Optional[pKaPredictor] = None,
    ) -> None:
        self.processor = processor or MoleculeProcessor()
        self.extractor = extractor or FeatureExtractor()
        self.predictor = predictor or pKaPredictor(self.extractor)

    def run(
        self,
        smiles_list: Sequence[str],
        ids: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Predict pKa for a list of SMILES and return a tidy DataFrame."""
        if ids is None:
            ids = [f"mol_{i}" for i in range(len(smiles_list))]
        if len(ids) != len(smiles_list):
            raise ValueError("ids and smiles_list must be the same length")

        all_rows: List[dict] = []
        for mol_id, smiles in zip(ids, smiles_list):
            record = self.processor.process(smiles, mol_id)
            if not record.is_valid:
                logger.warning("Skipping %s (%s): %s", mol_id, smiles, record.error)
            all_rows.extend(self.predictor.predict_molecule(record))

        return pd.DataFrame(all_rows)

    def run_csv(
        self,
        path: str,
        smiles_column: str = "smiles",
        id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read a CSV, predict, and return a DataFrame.

        The CSV must contain a SMILES column. An ID column is optional; if
        absent, sequential IDs are generated.
        """
        df = pd.read_csv(path)
        if smiles_column not in df.columns:
            raise KeyError(f"Column '{smiles_column}' not found in {path}")
        ids = df[id_column].astype(str).tolist() if id_column else None
        return self.run(df[smiles_column].tolist(), ids)


# --------------------------------------------------------------------------- #
#  Demo / CLI entry-point
# --------------------------------------------------------------------------- #
def _demo() -> pd.DataFrame:
    """Run a small illustrative batch (works without any trained model)."""
    demo_smiles = {
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",          # acidic (COOH)
        "lidocaine": "CCN(CC)CC(=O)Nc1c(C)cccc1C",     # basic (tertiary amine)
        "ciprofloxacin": "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",  # amphoteric
        "sodium_benzoate_salt": "[Na+].O=C([O-])c1ccccc1",  # salt handling
        "invalid": "this_is_not_smiles",              # error handling
    }
    pipeline = PkaPipeline()
    result = pipeline.run(
        list(demo_smiles.values()), list(demo_smiles.keys())
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate pKa from SMILES.")
    parser.add_argument("--csv", help="Input CSV path")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--acidic-model", default=None, help="joblib model path")
    parser.add_argument("--basic-model", default=None, help="joblib model path")
    parser.add_argument("--out", default=None, help="Output CSV path")
    args = parser.parse_args()

    extractor = FeatureExtractor()
    predictor = pKaPredictor.load(
        extractor, args.acidic_model, args.basic_model
    ) if (args.acidic_model or args.basic_model) else pKaPredictor(extractor)
    pipe = PkaPipeline(extractor=extractor, predictor=predictor)

    if args.csv:
        out_df = pipe.run_csv(args.csv, args.smiles_col, args.id_col)
    else:
        logger.info("No --csv supplied; running built-in demo.")
        out_df = _demo()

    pd.set_option("display.max_columns", None, "display.width", 160)
    print(out_df.to_string(index=False))
    if args.out:
        out_df.to_csv(args.out, index=False)
        logger.info("Wrote %s", args.out)
