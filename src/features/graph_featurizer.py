"""
Graph Neural Networks feature extraction for molecular pIC50 prediction.
RTX3060 optimized implementation with PyTorch Geometric.
"""

import hashlib
import logging

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from torch_geometric.data import Batch, Data

from ..utils.cache import FeatureCache


class MolecularGraphFeaturizer:
    """Graph-based molecular feature extraction for GNN models.

    Converts molecules to graph representations with node and edge features
    optimized for RTX3060 memory constraints.
    """

    def __init__(self, cache_dir: str = ".cache"):
        """Initialize the graph featurizer.

        Args:
            cache_dir: Directory for caching graph features
        """
        self.cache = FeatureCache(cache_dir)
        self.logger = logging.getLogger(__name__)

        # Atom feature dimensions (optimized for RTX3060)
        self.atom_feature_dims = {
            "atomic_num": 118,  # Periodic table elements
            "degree": 10,  # Max degree in organic molecules
            "formal_charge": 5,  # -2 to +2
            "hybridization": 6,  # sp, sp2, sp3, sp3d, sp3d2, other
            "num_h": 5,  # 0-4 hydrogens
            "aromatic": 2,  # boolean
            "chirality": 2,  # boolean
            "in_ring": 2,  # boolean
        }

        # Edge feature dimensions
        self.edge_feature_dims = {
            "bond_type": 4,  # single, double, triple, aromatic
            "conjugated": 2,  # boolean
            "in_ring": 2,  # boolean
            "stereo": 6,  # stereo configuration
        }

        # Calculate total dimensions
        self.node_feature_dim = sum(self.atom_feature_dims.values())
        self.edge_feature_dim = sum(self.edge_feature_dims.values())

        self.logger.info(
            f"Graph featurizer initialized: node_dim={self.node_feature_dim}, edge_dim={self.edge_feature_dim}"
        )

    def _get_smiles_hash(self, smiles: str) -> str:
        """Get MD5 hash of SMILES string for caching.

        Args:
            smiles: SMILES string

        Returns:
            MD5 hash string
        """
        return hashlib.md5(f"graph_{smiles}".encode()).hexdigest()

    def _get_atom_features(self, atom: Chem.Atom) -> torch.Tensor:
        """Extract atom features for GNN.

        Args:
            atom: RDKit atom object

        Returns:
            One-hot encoded atom features
        """
        features = []

        # Atomic number (one-hot)
        atomic_num = atom.GetAtomicNum()
        atomic_onehot = torch.zeros(self.atom_feature_dims["atomic_num"])
        if atomic_num < self.atom_feature_dims["atomic_num"]:
            atomic_onehot[atomic_num] = 1.0
        features.append(atomic_onehot)

        # Degree (one-hot)
        degree = min(atom.GetDegree(), self.atom_feature_dims["degree"] - 1)
        degree_onehot = torch.zeros(self.atom_feature_dims["degree"])
        degree_onehot[degree] = 1.0
        features.append(degree_onehot)

        # Formal charge (one-hot)
        formal_charge = atom.GetFormalCharge() + 2  # Shift to 0-4 range
        formal_charge = max(0, min(formal_charge, self.atom_feature_dims["formal_charge"] - 1))
        charge_onehot = torch.zeros(self.atom_feature_dims["formal_charge"])
        charge_onehot[formal_charge] = 1.0
        features.append(charge_onehot)

        # Hybridization (one-hot)
        hybridization = atom.GetHybridization()
        hybrid_map = {
            Chem.HybridizationType.SP: 0,
            Chem.HybridizationType.SP2: 1,
            Chem.HybridizationType.SP3: 2,
            Chem.HybridizationType.SP3D: 3,
            Chem.HybridizationType.SP3D2: 4,
        }
        hybrid_idx = hybrid_map.get(hybridization, 5)  # other = 5
        hybrid_onehot = torch.zeros(self.atom_feature_dims["hybridization"])
        hybrid_onehot[hybrid_idx] = 1.0
        features.append(hybrid_onehot)

        # Number of hydrogens (one-hot)
        num_h = min(atom.GetTotalNumHs(), self.atom_feature_dims["num_h"] - 1)
        num_h_onehot = torch.zeros(self.atom_feature_dims["num_h"])
        num_h_onehot[num_h] = 1.0
        features.append(num_h_onehot)

        # Aromatic (one-hot)
        aromatic_onehot = torch.zeros(self.atom_feature_dims["aromatic"])
        aromatic_onehot[int(atom.GetIsAromatic())] = 1.0
        features.append(aromatic_onehot)

        # Chirality (one-hot)
        chirality_onehot = torch.zeros(self.atom_feature_dims["chirality"])
        chirality_onehot[int(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)] = 1.0
        features.append(chirality_onehot)

        # In ring (one-hot)
        in_ring_onehot = torch.zeros(self.atom_feature_dims["in_ring"])
        in_ring_onehot[int(atom.IsInRing())] = 1.0
        features.append(in_ring_onehot)

        return torch.cat(features, dim=0)

    def _get_bond_features(self, bond: Chem.Bond) -> torch.Tensor:
        """Extract bond features for GNN.

        Args:
            bond: RDKit bond object

        Returns:
            One-hot encoded bond features
        """
        features = []

        # Bond type (one-hot)
        bond_type = bond.GetBondType()
        bond_type_map = {
            BondType.SINGLE: 0,
            BondType.DOUBLE: 1,
            BondType.TRIPLE: 2,
            BondType.AROMATIC: 3,
        }
        bond_type_idx = bond_type_map.get(bond_type, 0)
        bond_type_onehot = torch.zeros(self.edge_feature_dims["bond_type"])
        bond_type_onehot[bond_type_idx] = 1.0
        features.append(bond_type_onehot)

        # Conjugated (one-hot)
        conjugated_onehot = torch.zeros(self.edge_feature_dims["conjugated"])
        conjugated_onehot[int(bond.GetIsConjugated())] = 1.0
        features.append(conjugated_onehot)

        # In ring (one-hot)
        in_ring_onehot = torch.zeros(self.edge_feature_dims["in_ring"])
        in_ring_onehot[int(bond.IsInRing())] = 1.0
        features.append(in_ring_onehot)

        # Stereo (one-hot)
        stereo = bond.GetStereo()
        stereo_map = {
            Chem.BondStereo.STEREONONE: 0,
            Chem.BondStereo.STEREOANY: 1,
            Chem.BondStereo.STEREOZ: 2,
            Chem.BondStereo.STEREOE: 3,
            Chem.BondStereo.STEREOCIS: 4,
            Chem.BondStereo.STEREOTRANS: 5,
        }
        stereo_idx = stereo_map.get(stereo, 0)
        stereo_onehot = torch.zeros(self.edge_feature_dims["stereo"])
        stereo_onehot[stereo_idx] = 1.0
        features.append(stereo_onehot)

        return torch.cat(features, dim=0)

    def smiles_to_graph(self, smiles: str) -> Data | None:
        """Convert SMILES to PyTorch Geometric graph.

        Args:
            smiles: SMILES string

        Returns:
            PyTorch Geometric Data object or None if conversion fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.logger.warning(f"Invalid SMILES: {smiles}")
                return None

            # Add hydrogens for complete graph
            mol = Chem.AddHs(mol)

            # Get atom features
            atom_features = []
            for atom in mol.GetAtoms():
                features = self._get_atom_features(atom)
                atom_features.append(features)

            if not atom_features:
                return None

            x = torch.stack(atom_features, dim=0)

            # Get edge indices and features
            edge_indices = []
            edge_features = []

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                # Add both directions for undirected graph
                edge_indices.extend([[i, j], [j, i]])

                # Get bond features
                bond_features = self._get_bond_features(bond)
                edge_features.extend([bond_features, bond_features])

            if not edge_indices:
                # Single atom molecule
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, self.edge_feature_dim), dtype=torch.float)
            else:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.stack(edge_features, dim=0)

            # Create PyTorch Geometric Data object
            data = Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles, num_nodes=x.size(0)
            )

            return data

        except Exception as e:
            self.logger.error(f"Graph conversion error for {smiles}: {e}")
            return None

    def calculate_graph_features(self, smiles: str) -> Data | None:
        """Calculate graph features with caching.

        Args:
            smiles: SMILES string

        Returns:
            PyTorch Geometric Data object or None if calculation fails
        """
        # Check cache first
        smiles_hash = self._get_smiles_hash(smiles)
        cached_data = self.cache.get(smiles_hash)
        if cached_data is not None:
            # Reconstruct Data object from cached numpy arrays
            return self._reconstruct_from_cache(cached_data, smiles)

        # Calculate graph features
        data = self.smiles_to_graph(smiles)
        if data is not None:
            # Cache the graph data
            self._cache_graph_data(smiles_hash, data)

        return data

    def _cache_graph_data(self, smiles_hash: str, data: Data) -> None:
        """Cache graph data as numpy arrays.

        Args:
            smiles_hash: Hash of SMILES string
            data: PyTorch Geometric Data object
        """
        try:
            cache_data = {
                "x": data.x.numpy(),
                "edge_index": data.edge_index.numpy(),
                "edge_attr": data.edge_attr.numpy(),
                "num_nodes": data.num_nodes,
            }
            self.cache.save(smiles_hash, cache_data)
        except Exception as e:
            self.logger.warning(f"Failed to cache graph data: {e}")

    def _reconstruct_from_cache(self, cache_data: dict, smiles: str) -> Data:
        """Reconstruct Data object from cached numpy arrays.

        Args:
            cache_data: Cached data dictionary
            smiles: SMILES string

        Returns:
            Reconstructed PyTorch Geometric Data object
        """
        try:
            data = Data(
                x=torch.from_numpy(cache_data["x"]),
                edge_index=torch.from_numpy(cache_data["edge_index"]),
                edge_attr=torch.from_numpy(cache_data["edge_attr"]),
                smiles=smiles,
                num_nodes=cache_data["num_nodes"],
            )
            return data
        except Exception as e:
            self.logger.warning(f"Failed to reconstruct from cache: {e}")
            return None

    def calculate_batch_graph_features(
        self, smiles_list: list[str]
    ) -> tuple[list[Data], list[int]]:
        """Calculate graph features for a batch of SMILES.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Tuple of (graph_data_list, valid_indices)
        """
        graph_data_list = []
        valid_indices = []

        for i, smiles in enumerate(smiles_list):
            data = self.calculate_graph_features(smiles)
            if data is not None:
                graph_data_list.append(data)
                valid_indices.append(i)
            else:
                self.logger.warning(f"Failed to calculate graph features for SMILES {i}: {smiles}")

        if not graph_data_list:
            raise ValueError("No valid graph features could be calculated")

        self.logger.info(
            f"Calculated graph features for {len(graph_data_list)}/{len(smiles_list)} molecules"
        )
        return graph_data_list, valid_indices

    def create_batch(self, graph_data_list: list[Data]) -> Batch:
        """Create a batch from list of graph data.

        Args:
            graph_data_list: List of PyTorch Geometric Data objects

        Returns:
            Batched PyTorch Geometric Data object
        """
        return Batch.from_data_list(graph_data_list)

    def get_feature_dims(self) -> dict[str, int]:
        """Get feature dimensions for model initialization.

        Returns:
            Dictionary with node and edge feature dimensions
        """
        return {
            "node_feature_dim": self.node_feature_dim,
            "edge_feature_dim": self.edge_feature_dim,
        }

    def get_graph_statistics(self, graph_data_list: list[Data]) -> dict[str, float]:
        """Get statistics about the graph dataset.

        Args:
            graph_data_list: List of graph data objects

        Returns:
            Dictionary with graph statistics
        """
        if not graph_data_list:
            return {}

        num_nodes_list = [data.num_nodes for data in graph_data_list]
        num_edges_list = [data.edge_index.size(1) for data in graph_data_list]

        return {
            "num_graphs": len(graph_data_list),
            "avg_num_nodes": np.mean(num_nodes_list),
            "std_num_nodes": np.std(num_nodes_list),
            "max_num_nodes": np.max(num_nodes_list),
            "min_num_nodes": np.min(num_nodes_list),
            "avg_num_edges": np.mean(num_edges_list),
            "std_num_edges": np.std(num_edges_list),
            "max_num_edges": np.max(num_edges_list),
            "min_num_edges": np.min(num_edges_list),
        }
