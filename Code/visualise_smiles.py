from rdkit import Chem
from rdkit.Chem import Draw


def visualize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Draw.MolToImage(mol)
        mol_size = mol.GetNumAtoms() * 50
        img = Draw.MolToImage(mol, size=(mol_size, mol_size))
        img.show()
    else:
        print("Invalid SMILES string")


# Example usage:
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
visualize_smiles(smiles)
