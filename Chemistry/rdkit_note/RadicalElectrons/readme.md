

```python
def neutralizeRadicals(mol):
    for a in mol.GetAtoms():
        if a.GetNumRadicalElectrons()>0:
            a.SetNumRadicalElectrons(0)
```

```python
import rdkit
from rdkit import Chem
smiles = "C[C@H]1CCC[N@@H+]([C][C][C](C(=O)C([O])=O)[C@]2([O])[C]3[C][C]c4oc(Oc5cccc([C@]6(c7nncn7C)C[C@@H](C)C6)c5)c([C]([C][N])[C]=O)c4[C]32)C1"

mol = Chem.MolFromSmiles(smiles)
```

```python
neutralizeRadicals(mol)
Chem.MolToSmiles(mol)
```