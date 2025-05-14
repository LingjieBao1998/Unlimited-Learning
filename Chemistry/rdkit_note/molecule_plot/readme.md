

## Set `AtomNum` in molecule
```
def annotating_atoms(mol):
    [atom.SetAtomMapNum(_+1) for _, atom in enumerate(mol.GetAtoms())]
``` 


## Remove `AtomNum` in molecule
```
def clear_annotating_atoms(mol):
    [atom.SetAtomMapNum(0) for _, atom in enumerate(mol.GetAtoms())]
```


