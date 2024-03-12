import requests
import csv
import re


def parse_sdf(sdf_text):
    # Split the SDF text into lines
    lines = sdf_text.strip().split('\n')

    # Initialize a list to store the parsed data
    parsed_coords = []
    parsed_bonds = []
    parsed_charges = []
    # Flag to indicate parsing the partial charges section
    parsing_charges = False

    # Parse the SDF text line by line
    index = 1
    for line in lines:
        # Check if it's an atom line
        if len(line) >= 60:
            # Extract atom type and coordinates
            x = float(line[:10])
            y = float(line[10:20])
            z = float(line[20:30])
            atom_type = line[31:34].strip()
            # Append parsed data to the list
            parsed_coords.append((index, atom_type, x, y, z))
            index += 1

        elif 20 <= len(line) <= 22:
            try:
                # Extract bond information
                atom1_idx = int(line[:3])
                atom2_idx = int(line[4:7])
                bond_type = int(line[8])
                parsed_bonds.append((atom1_idx, atom2_idx, bond_type))
            except ValueError:
                # Skip lines that don't start with digit characters
                continue

        elif parsing_charges:
            if line.startswith('>'):
                parsing_charges = False
            else:
                try:
                    # Extract atom index and partial charge
                    atom_idx, charge = map(float, line.strip().split())
                    parsed_charges.append((int(atom_idx), charge))
                except ValueError:
                    # Skip lines that don't contain valid charge data
                    continue
        elif line.startswith('> <PUBCHEM_MMFF94_PARTIAL_CHARGES>'):
            # Start parsing partial charges
            parsing_charges = True

    return parsed_coords, parsed_bonds, parsed_charges


def get_compound_info(cid):
    # Retrieve Canonical SMILES
    smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    smiles_response = requests.get(smiles_url)
    canonical_smiles = None
    parsed_coords = None
    parsed_bonds = None
    parsed_charges = None
    if smiles_response.status_code == 200:
        smiles_data = smiles_response.json()
        if 'PropertyTable' in smiles_data:
            smiles_property_table = smiles_data['PropertyTable']
            if 'Properties' in smiles_property_table and len(smiles_property_table['Properties']) > 0:
                canonical_smiles = smiles_property_table['Properties'][0]['CanonicalSMILES']

    # Retrieve SDF file
    try:
        sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d"
        sdf_response = requests.get(sdf_url)
        parsed_info = None, None
        if sdf_response.status_code == 200:
            sdf_text = sdf_response.text
            # Parse SDF text to extract atoms, bonds, coordinates, and charge
            parsed_coords, parsed_bonds, parsed_charges = parse_sdf(sdf_text)
    except:
        return None
    return canonical_smiles, parsed_coords, parsed_bonds, parsed_charges


def write_compound_info(filename, output_filename):
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames[:]

        # Find the index of the "STITCH 1" column
        stitch1_index = fieldnames.index('# STITCH 1')
        # Insert new fieldnames to the immediate right of "STITCH 1"
        fieldnames.insert(stitch1_index + 1, 'C1 SMILES')
        fieldnames.insert(stitch1_index + 2, 'C1 Coords')
        fieldnames.insert(stitch1_index + 3, 'C1 Bonds')
        fieldnames.insert(stitch1_index + 4, 'C1 Charges')

        # Find the index of the "STITCH 2" column
        stitch2_index = fieldnames.index('STITCH 2')
        # Insert new fieldnames to the immediate right of "STITCH 2"
        fieldnames.insert(stitch2_index + 1, 'C2 SMILES')
        fieldnames.insert(stitch2_index + 2, 'C2 Coords')
        fieldnames.insert(stitch2_index + 3, 'C2 Bonds')
        fieldnames.insert(stitch2_index + 4, 'C2 Charges')

        # DECIDES WHICH ROW OF CSV TO START READING FROM
        start_row = 1014624

        with open(output_filename, mode='a', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            # Write the header row only if the file is empty
            if output_file.tell() == 0:
                writer.writeheader()

            last_stitch1_id = None
            last_stitch2_id = None

            # Iterate over the first X rows
            for index, row in enumerate(reader):
                if(index < start_row):
                    continue

                stitch1_id = row['# STITCH 1']

                # Gets the compound ID of varying length, removes trailing 0's
                compound_id1 = re.search(r'\d+', stitch1_id).group()
                compound_id1 = str(int(compound_id1))

                # Talk about how much time I saved by adding this check in, so that it reduces API requests
                if stitch1_id != last_stitch1_id:
                    stitch1_canonical_smiles, stitch1_parsed_coords, stitch1_parsed_bonds, stitch1_parsed_charges = get_compound_info(compound_id1)
                    if stitch1_canonical_smiles is None:
                        print("No data found for", stitch1_id)
                        continue
                    else:
                        last_stitch1_id = stitch1_id
                        print("New Stitch1 Request", stitch1_id)

                # Insert the information to the new columns
                row['C1 SMILES'] = stitch1_canonical_smiles
                row['C1 Coords'] = stitch1_parsed_coords
                row['C1 Bonds'] = stitch1_parsed_bonds
                row['C1 Charges'] = stitch1_parsed_charges

                stitch2_id = row['STITCH 2']

                # Gets the compound ID of varying length, removes trailing 0's
                compound_id2 = re.search(r'\d+', stitch2_id).group()
                compound_id2 = str(int(compound_id2))

                if stitch2_id != last_stitch2_id:
                    stitch2_canonical_smiles, stitch2_parsed_coords, stitch2_parsed_bonds, stitch2_parsed_charges = get_compound_info(compound_id2)
                    if stitch2_canonical_smiles is None:
                        print("No data found for", stitch2_id)
                        continue
                    else:
                        last_stitch2_id = stitch2_id
                        print("New Stitch2 Request", stitch2_id)

                # Insert the information to the new columns
                row['C2 SMILES'] = stitch2_canonical_smiles
                row['C2 Coords'] = stitch2_parsed_coords
                row['C2 Bonds'] = stitch2_parsed_bonds
                row['C2 Charges'] = stitch2_parsed_charges

                writer.writerow(row)
    print("File complete!")



# Testing with Aspirin:
canonical_smiles, parsed_coords, parsed_bonds, parsed_charges = get_compound_info(2244)
print("Aspirin Canonical SMILES:", canonical_smiles)
print("Aspirin Atom Info:", parsed_coords)
print("Aspirin Bond Info:", parsed_bonds)
print("Paracetamol Charge Info:", parsed_charges)

print("\n")

# Testing with Paracetamol:
canonical_smiles, parsed_coords, parsed_bonds, parsed_charges = get_compound_info(1983)
print("[Paracetamol Canonical SMILES:", canonical_smiles)
print("Paracetamol Atom Info:", parsed_coords)
print("Paracetamol Bond Info:", parsed_bonds)
print("Paracetamol Charge Info:", parsed_charges)

print("\n")

write_compound_info("ChChSe-Decagon_polypharmacy/ChChSe-Decagon_polypharmacy.csv", "ChChSe-Decagon_polypharmacy/cleanedData.csv")
