
# Enhancing JT-VAE with Torsional Angles

## **Background**
The **Junction Tree Variational Autoencoder (JT-VAE)** is a widely used generative model for molecular design. However, it primarily operates on **2D molecular graphs**, encoding molecules as junction trees composed of molecular substructures without explicitly considering **torsional angles**. Since **torsional flexibility** significantly impacts **molecular stability, bioactivity, and ligand-receptor interactions**, extending JT-VAE to incorporate these features can enhance molecular generation accuracy.

## **Why Include Torsional Angles?**
- **JT-VAE Limitations**: It does not account for **3D molecular conformations** or **torsional flexibility**.
- **Impact on Molecular Stability**: Many drug molecules require correct **torsional states** to fit binding sites.
- **More Accurate Molecular Generation**: Including torsional angles ensures the model generates **realistic and bioactive molecules**.

## **Modification: Incorporating Torsional Angles in JT-VAE**
### **1. Precompute Torsional Angles from Conformers**
Using **RDKit**, we extract **dihedral angles** for all **rotatable bonds**:

```python
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

def compute_torsional_angles(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=10, params=AllChem.ETKDGv3())
    torsion_angles = []

    for bond in mol.GetBonds():
        if bond.IsRotor():
            atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            neighbors1 = [n.GetIdx() for n in mol.GetAtomWithIdx(atom1).GetNeighbors() if n.GetIdx() != atom2]
            neighbors2 = [n.GetIdx() for n in mol.GetAtomWithIdx(atom2).GetNeighbors() if n.GetIdx() != atom1]

            if len(neighbors1) == 0 or len(neighbors2) == 0:
                continue

            atom3, atom4 = neighbors1[0], neighbors2[0]
            angles = [rdMolTransforms.GetDihedralDeg(conf, atom3, atom1, atom2, atom4) for conf in mol.GetConformers()]
            torsion_angles.append(sum(angles) / len(angles))  # Averaging angles

    return torsion_angles
```

### **2. Modify Molecular Graph Representation in DGL**
Torsional angles are added as **bond features** in DGL:

```python
import dgl
import torch
from dgl.data.chem import mol_to_graph

def mol_to_dgl_with_torsion(mol):
    torsion_angles = compute_torsional_angles(mol)
    dgl_graph = mol_to_graph(mol)

    bond_features = []
    bond_count = 0
    for bond in mol.GetBonds():
        features = [bond.GetBondTypeAsDouble()]
        if bond.IsRotor():
            features.append(torsion_angles[bond_count])
            bond_count += 1
        else:
            features.append(0.0)  # Non-rotatable bonds get 0

        bond_features.append(features)

    dgl_graph.edata['torsion_angle'] = torch.tensor(bond_features, dtype=torch.float)
    return dgl_graph
```

### **3. Modify JT-VAE Encoder to Process Torsional Angles**
We modify the **GNN-based encoder** to include torsional angles:

```python
import torch.nn as nn
import dgl.function as fn

class JT_VAE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(JT_VAE_Encoder, self).__init__()
        self.node_fc = nn.Linear(input_dim, hidden_dim)
        self.edge_fc = nn.Linear(2, hidden_dim)  # Includes torsional angles
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, g):
        h = self.node_fc(g.ndata['h'])
        e = self.edge_fc(g.edata['torsion_angle'])

        g.ndata['h'] = h
        g.edata['e'] = e
        g.update_all(fn.u_add_e('h', 'e', 'm'), fn.mean('m', 'h'))

        h = self.gru(h.unsqueeze(0))[0].squeeze(0)
        return h
```

### **4. Modify Decoder for Torsional-Aware Molecular Generation**
We adjust the **decoder** to reconstruct **torsional angles**:

```python
class JT_VAE_Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(JT_VAE_Decoder, self).__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.bond_predictor = nn.Linear(hidden_dim, output_dim)  # Predicts torsional angles

    def forward(self, h):
        h, _ = self.gru(h.unsqueeze(0))
        bond_features = self.bond_predictor(h.squeeze(0))
        return bond_features
```

### **5. Training with Torsion-Aware Loss Function**
To optimize torsion-aware molecular generation:

```python
def torsion_aware_loss(pred_bonds, true_bonds, pred_torsions, true_torsions):
    bond_loss = nn.CrossEntropyLoss()(pred_bonds, true_bonds)
    torsion_loss = nn.MSELoss()(pred_torsions, true_torsions)  # Torsional angle regression
    return bond_loss + torsion_loss
```

### **Final Notes**
- **This modification enhances JT-VAE by making it torsion-aware**, improving **3D molecular fidelity**.
- **Compared to standard JT-VAE**, this version generates molecules with **realistic torsional flexibility**, making it more useful for **drug discovery and structure-based design**.

