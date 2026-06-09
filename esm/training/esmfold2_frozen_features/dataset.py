from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx
import numpy as np
import torch
from torch.utils.data import Dataset

from esm.models.esmfold2.constants import (
    MOL_TYPE_DNA,
    MOL_TYPE_NONPOLYMER,
    MOL_TYPE_RNA,
)
from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
from esm.models.esmfold2.types import ProteinInput, StructurePredictionInput
from esm.utils import residue_constants
from esm.utils.structure.mmcif_parsing import MmcifWrapper
from esm.utils.structure.protein_chain import ProteinChain

SAMPLE_MANIFEST_ROWS: tuple[tuple[str, ...], ...] = (
    (
        "nanobody_001",
        "1MEL",
        "https://files.rcsb.org/download/1MEL.cif",
        "A",
        "A",
        "label_asym_id_first_then_auth_asym_id",
        "(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",
        "monomer_vhh_train",
        "train",
    ),
    (
        "nanobody_002",
        "1MEL",
        "https://files.rcsb.org/download/1MEL.cif",
        "B",
        "B",
        "label_asym_id_first_then_auth_asym_id",
        "(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",
        "monomer_vhh_train",
        "train",
    ),
    (
        "nanobody_003",
        "1I3V",
        "https://files.rcsb.org/download/1I3V.cif",
        "A",
        "A",
        "label_asym_id_first_then_auth_asym_id",
        "(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",
        "monomer_vhh_train",
        "train",
    ),
    (
        "nanobody_004",
        "1I3V",
        "https://files.rcsb.org/download/1I3V.cif",
        "B",
        "B",
        "label_asym_id_first_then_auth_asym_id",
        "(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",
        "monomer_vhh_train",
        "train",
    ),
    (
        "nanobody_005",
        "6JB9",
        "https://files.rcsb.org/download/6JB9.cif",
        "A",
        "A",
        "label_asym_id_first_then_auth_asym_id",
        "(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",
        "monomer_vhh_train",
        "train",
    ),
    (
        "nanobody_006",
        "5U64",
        "https://files.rcsb.org/download/5U64.cif",
        "A",
        "B",
        "label_asym_id_first_then_auth_asym_id",
        "(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",
        "monomer_vhh_train",
        "train",
    ),
    (
        "nanobody_007",
        "4KRN",
        "https://files.rcsb.org/download/4KRN.cif",
        "A",
        "A",
        "label_asym_id_first_then_auth_asym_id",
        "(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",
        "monomer_vhh_train",
        "train",
    ),
    (
        "nanobody_008",
        "4KRL",
        "https://files.rcsb.org/download/4KRL.cif",
        "A",
        "B",
        "label_asym_id_first_then_auth_asym_id",
        "(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",
        "monomer_vhh_train",
        "train",
    ),
    (
        "nanobody_009",
        "4KRO",
        "https://files.rcsb.org/download/4KRO.cif",
        "B",
        "B",
        "label_asym_id_first_then_auth_asym_id",
        "(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",
        "monomer_vhh_train",
        "train",
    ),
    (
        "nanobody_010",
        "4KRP",
        "https://files.rcsb.org/download/4KRP.cif",
        "B",
        "B",
        "label_asym_id_first_then_auth_asym_id",
        "(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",
        "monomer_vhh_train",
        "train",
    ),
)

MANIFEST_HEADER = (
    "example_id",
    "pdb_id",
    "mmcif_url",
    "label_asym_id",
    "auth_asym_id",
    "chain_match_policy",
    "expected_entity_name_regex",
    "task",
    "split",
)

SAMPLE_MANIFEST_TEXT = """example_id,pdb_id,mmcif_url,label_asym_id,auth_asym_id,chain_match_policy,expected_entity_name_regex,task,split
nanobody_001,1MEL,https://files.rcsb.org/download/1MEL.cif,A,A,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_002,1MEL,https://files.rcsb.org/download/1MEL.cif,B,B,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_003,1I3V,https://files.rcsb.org/download/1I3V.cif,A,A,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_004,1I3V,https://files.rcsb.org/download/1I3V.cif,B,B,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_005,6JB9,https://files.rcsb.org/download/6JB9.cif,A,A,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_006,5U64,https://files.rcsb.org/download/5U64.cif,A,B,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_007,4KRN,https://files.rcsb.org/download/4KRN.cif,A,A,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_008,4KRL,https://files.rcsb.org/download/4KRL.cif,A,B,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_009,4KRO,https://files.rcsb.org/download/4KRO.cif,B,B,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_010,4KRP,https://files.rcsb.org/download/4KRP.cif,B,B,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
"""


@dataclass(frozen=True)
class ManifestRow:
    example_id: str
    pdb_id: str
    mmcif_url: str
    label_asym_id: str
    auth_asym_id: str
    chain_match_policy: str
    expected_entity_name_regex: str
    task: str
    split: str

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "ManifestRow":
        missing = [name for name in MANIFEST_HEADER if not row.get(name)]
        if missing:
            raise ValueError(f"manifest row is missing required fields: {missing}")
        return cls(**{name: row[name] for name in MANIFEST_HEADER})


def write_sample_manifest(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SAMPLE_MANIFEST_TEXT)
    return path


def read_manifest(path: str | Path) -> list[ManifestRow]:
    with Path(path).open(newline="") as handle:
        rows = [ManifestRow.from_dict(row) for row in csv.DictReader(handle)]
    return rows


def load_cache_index(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    index_path = Path(path)
    records: dict[str, dict[str, Any]] = {}
    with index_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            example_id = record["example_id"]
            if example_id in records:
                raise ValueError(f"duplicate cache record for {example_id}")
            hidden_path = Path(record["hidden_states_path"])
            if not hidden_path.is_absolute():
                hidden_path = (index_path.parent / hidden_path).resolve()
            record["hidden_states_path"] = str(hidden_path)
            records[example_id] = record
    return records


def download_mmcif(row: ManifestRow, mmcif_dir: str | Path) -> Path:
    mmcif_dir = Path(mmcif_dir)
    mmcif_dir.mkdir(parents=True, exist_ok=True)
    path = mmcif_dir / f"{row.pdb_id.upper()}.cif"
    if path.exists() and path.stat().st_size > 0:
        return path
    response = httpx.get(row.mmcif_url, follow_redirects=True, timeout=60.0)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def _category_array(mmcif: MmcifWrapper, category: str, column: str) -> list[str]:
    if mmcif.raw is None:
        return []
    block = mmcif.raw.block
    if category not in block:
        return []
    cat = block[category]
    if column not in cat:
        return []
    return list(cat[column].as_array(str))


def _entity_descriptions(mmcif: MmcifWrapper) -> dict[str, str]:
    ids = _category_array(mmcif, "entity", "id")
    descriptions = _category_array(mmcif, "entity", "pdbx_description")
    return {entity_id: desc for entity_id, desc in zip(ids, descriptions)}


def _label_to_entity(mmcif: MmcifWrapper) -> dict[str, str]:
    labels = _category_array(mmcif, "struct_asym", "id")
    entities = _category_array(mmcif, "struct_asym", "entity_id")
    mapping = {label: entity for label, entity in zip(labels, entities)}
    if mapping:
        return mapping
    labels = _category_array(mmcif, "atom_site", "label_asym_id")
    entities = _category_array(mmcif, "atom_site", "label_entity_id")
    for label, entity in zip(labels, entities):
        mapping.setdefault(label, entity)
    return mapping


def _label_to_auth(mmcif: MmcifWrapper) -> dict[str, str]:
    labels = _category_array(mmcif, "pdbx_poly_seq_scheme", "asym_id")
    auths = _category_array(mmcif, "pdbx_poly_seq_scheme", "pdb_strand_id")
    mapping = {label: auth for label, auth in zip(labels, auths) if auth not in {"?", "."}}
    if mapping:
        return mapping
    labels = _category_array(mmcif, "atom_site", "label_asym_id")
    auths = _category_array(mmcif, "atom_site", "auth_asym_id")
    for label, auth in zip(labels, auths):
        if auth not in {"?", "."}:
            mapping.setdefault(label, auth)
    return mapping


def validate_entity_name(mmcif: MmcifWrapper, row: ManifestRow) -> str:
    label_entity = _label_to_entity(mmcif).get(row.label_asym_id)
    descriptions = _entity_descriptions(mmcif)
    description = descriptions.get(label_entity or "", "")
    if not description:
        raise ValueError(
            f"{row.example_id}: could not find entity description for label chain "
            f"{row.label_asym_id}"
        )
    if re.search(row.expected_entity_name_regex, description) is None:
        raise ValueError(
            f"{row.example_id}: entity description {description!r} does not match "
            f"{row.expected_entity_name_regex!r}"
        )
    return description


def _chain_candidates(mmcif: MmcifWrapper, row: ManifestRow) -> list[str]:
    if row.chain_match_policy != "label_asym_id_first_then_auth_asym_id":
        raise ValueError(
            f"{row.example_id}: unsupported chain_match_policy "
            f"{row.chain_match_policy!r}"
        )
    label_to_auth = _label_to_auth(mmcif)
    candidates = [
        row.label_asym_id,
        label_to_auth.get(row.label_asym_id, ""),
        row.auth_asym_id,
    ]
    result: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in result:
            result.append(candidate)
    return result


def load_selected_chain(
    row: ManifestRow, mmcif_path: str | Path, *, keep_source: bool = True
) -> tuple[ProteinChain, MmcifWrapper, str]:
    mmcif = MmcifWrapper.read(mmcif_path, id=row.pdb_id)
    entity_description = validate_entity_name(mmcif, row)
    errors: list[str] = []
    for chain_id in _chain_candidates(mmcif, row):
        try:
            chain = ProteinChain.from_mmcif(
                mmcif, chain_id=chain_id, keep_source=keep_source
            )
        except Exception as exc:  # noqa: BLE001 - preserve all candidate failures
            errors.append(f"{chain_id}: {exc}")
            continue
        if len(chain.sequence) == 0 or not chain.atom37_mask.any():
            errors.append(f"{chain_id}: empty sequence or no resolved atoms")
            continue
        return chain, mmcif, entity_description
    raise ValueError(
        f"{row.example_id}: selected chain was not found; candidates="
        f"{_chain_candidates(mmcif, row)!r}; errors={errors}"
    )


def _decode_atom_name(chars: torch.Tensor) -> str:
    return "".join(chr(int(x) + 32) if int(x) != 0 else " " for x in chars).strip()


def build_features_from_chain(chain: ProteinChain) -> dict[str, torch.Tensor]:
    structure_input = StructurePredictionInput(
        sequences=[ProteinInput(id=chain.chain_id, sequence=chain.sequence, msa=None)]
    )
    features, _ = prepare_esmfold2_input(structure_input)

    atom_mask = features["atom_attention_mask"].bool()
    n_atoms = atom_mask.shape[0]
    gt_coords = torch.zeros(n_atoms, 3, dtype=torch.float32)
    gt_mask = torch.zeros(n_atoms, dtype=torch.bool)
    residue_index = features["residue_index"].long()
    atom_to_token = features["atom_to_token"].long()

    for atom_idx in atom_mask.nonzero(as_tuple=True)[0].tolist():
        token_idx = int(atom_to_token[atom_idx])
        res_idx = int(residue_index[token_idx])
        atom_name = _decode_atom_name(features["ref_atom_name_chars"][atom_idx])
        rc_idx = residue_constants.atom_order.get(atom_name)
        if rc_idx is None:
            continue
        if res_idx >= chain.atom37_mask.shape[0] or not chain.atom37_mask[res_idx, rc_idx]:
            continue
        coord = chain.atom37_positions[res_idx, rc_idx]
        if not np.isfinite(coord).all():
            continue
        gt_coords[atom_idx] = torch.from_numpy(coord.astype(np.float32))
        gt_mask[atom_idx] = True

    if gt_mask.any():
        centroid = gt_coords[gt_mask].mean(dim=0, keepdim=True)
        gt_coords[gt_mask] = gt_coords[gt_mask] - centroid

    mol_type_by_atom = features["mol_type"][atom_to_token.clamp_min(0)]
    atom_loss_weight = torch.ones(n_atoms, dtype=torch.float32)
    atom_loss_weight = torch.where(
        (mol_type_by_atom == MOL_TYPE_DNA) | (mol_type_by_atom == MOL_TYPE_RNA),
        torch.full_like(atom_loss_weight, 5.0),
        atom_loss_weight,
    )
    atom_loss_weight = torch.where(
        mol_type_by_atom == MOL_TYPE_NONPOLYMER,
        torch.full_like(atom_loss_weight, 10.0),
        atom_loss_weight,
    )
    atom_loss_weight = atom_loss_weight * atom_mask.float()

    features["gt_atom_coords"] = gt_coords
    features["gt_atom_mask"] = gt_mask
    features["gt_coords"] = gt_coords.unsqueeze(0)
    features["is_resolved"] = gt_mask
    features["atom_loss_weight"] = atom_loss_weight
    return features


def _load_hidden_states(record: dict[str, Any], sequence: str) -> torch.Tensor:
    hidden_states = torch.load(
        record["hidden_states_path"], map_location="cpu", weights_only=True
    )
    if isinstance(hidden_states, dict):
        hidden_states = hidden_states.get("lm_hidden_states", hidden_states.get("hidden_states"))
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError(f"cache file {record['hidden_states_path']} did not contain a tensor")
    if hidden_states.ndim != 3:
        raise ValueError(
            f"cached hidden states must have shape [L, N_layers+1, D], got "
            f"{tuple(hidden_states.shape)}"
        )
    if hidden_states.shape[0] != len(sequence):
        raise ValueError(
            f"cache length {hidden_states.shape[0]} does not match sequence length "
            f"{len(sequence)} for {record['example_id']}"
        )
    if record.get("sequence") and record["sequence"] != sequence:
        raise ValueError(f"cache sequence mismatch for {record['example_id']}")
    expected_shape = record.get("shape")
    if expected_shape is not None and list(hidden_states.shape) != list(expected_shape):
        raise ValueError(
            f"cache shape metadata {expected_shape} does not match tensor shape "
            f"{tuple(hidden_states.shape)} for {record['example_id']}"
        )
    expected_dtype = record.get("dtype")
    if expected_dtype is not None:
        dtype_name = str(hidden_states.dtype).replace("torch.", "")
        aliases = {"bf16": "bfloat16", "float": "float32"}
        normalized_actual = aliases.get(dtype_name, dtype_name)
        normalized_expected = aliases.get(str(expected_dtype), str(expected_dtype))
        if normalized_actual != normalized_expected:
            raise ValueError(
                f"cache dtype metadata {expected_dtype!r} does not match tensor dtype "
                f"{dtype_name!r} for {record['example_id']}"
            )
    return hidden_states


class ESMFold2FrozenFeatureDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        cache_index: str | Path | None = None,
        *,
        mmcif_dir: str | Path | None = None,
        require_cache: bool = True,
        download: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest)
        self.rows = read_manifest(self.manifest_path)
        self.cache_records = load_cache_index(cache_index)
        self.require_cache = require_cache
        self.download = download
        self.mmcif_dir = Path(mmcif_dir) if mmcif_dir else self.manifest_path.parent / "mmcif"

    def __len__(self) -> int:
        return len(self.rows)

    def _mmcif_path(self, row: ManifestRow) -> Path:
        path = self.mmcif_dir / f"{row.pdb_id.upper()}.cif"
        if path.exists():
            return path
        if not self.download:
            raise FileNotFoundError(f"missing mmCIF file: {path}")
        return download_mmcif(row, self.mmcif_dir)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        chain, mmcif, entity_description = load_selected_chain(row, self._mmcif_path(row))
        features = build_features_from_chain(chain)

        record = self.cache_records.get(row.example_id)
        if record is None and self.require_cache:
            raise KeyError(f"missing cache record for {row.example_id}")
        if record is not None:
            features["lm_hidden_states"] = _load_hidden_states(record, chain.sequence)

        features["example_id"] = row.example_id
        features["pdb_id"] = row.pdb_id
        features["sequence"] = chain.sequence
        features["selected_chain_id"] = chain.chain_id
        features["label_asym_id"] = row.label_asym_id
        features["auth_asym_id"] = row.auth_asym_id
        features["entity_description"] = entity_description
        resolution = mmcif.header.resolution
        features["resolution"] = torch.tensor(
            float("nan") if resolution is None else float(resolution), dtype=torch.float32
        )
        return features


def iter_unique_sequences(rows: Iterable[ManifestRow], mmcif_dir: str | Path) -> Iterable[tuple[ManifestRow, str]]:
    seen: set[str] = set()
    for row in rows:
        mmcif_path = download_mmcif(row, mmcif_dir)
        chain, _, _ = load_selected_chain(row, mmcif_path, keep_source=False)
        if chain.sequence in seen:
            continue
        seen.add(chain.sequence)
        yield row, chain.sequence
