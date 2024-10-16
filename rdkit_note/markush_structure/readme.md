
## introduction
`CXSMILES` (ChemAxon Extended SMILES) is an extended representation of SMILES by ChemAxon


it is mostly used for **polymers** and **groups** of compounds

basic format
```
SMILES_String |<feature1>,<feature2>,...|
```
The basic idea is to use this to write `features` between vertical bars "|" after the separator, and add additional information.

Features that can be written include `atomic coordinates`, `atom labels`, `coordination bonds`, `hydrogen bonds`, and configurations.

`CXSMILES` is used to represent `Markush` strucuture `Markush` stucture.

> ref: https://magattaca.hatenablog.com/entry/2020/11/23/114422


### represent `alias` in rdkit

```
from rdkit import rdBase, Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
print('rdkit version: ', rdBase.rdkitVersion)
# rdkit version:  2020.03.2

alanine_monomer = Chem.MolFromSmiles('C[C@H](N[*])C([*])=O |r,$;;;R1;;R2;$|')
alanine_monomer 
```

```
```