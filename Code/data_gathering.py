import math
import requests
import csv
import re


# Function to parse an SDF file
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

        # Check if it's a bond line
        elif 20 <= len(line) <= 22:
            try:
                # Extract bond information
                atom1_idx = int(line[:3])
                atom2_idx = int(line[4:7])
                bond_type = int(line[8])
                parsed_bonds.append((atom1_idx, atom2_idx, bond_type))
            except ValueError:
                # Skip lines that don't contain valid data
                continue

        # Check if it's a charge line
        elif parsing_charges:
            # Check to see if it's the end of the charge information
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
        # Check for the beginning of the charges information
        elif line.startswith('> <PUBCHEM_MMFF94_PARTIAL_CHARGES>'):
            # Start parsing partial charges
            parsing_charges = True

    return parsed_coords, parsed_bonds, parsed_charges


# Function to retrieve 3d compound information for a given ID
def get_compound_info(cid):
    # Set the URL for the PUG REST API
    smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    # Make a request to the API with that URL
    smiles_response = requests.get(smiles_url)
    canonical_smiles = None
    parsed_coords = None
    parsed_bonds = None
    parsed_charges = None
    # If there is a response then save the SMILES string as canonical_smiles
    if smiles_response.status_code == 200:
        smiles_data = smiles_response.json()
        if 'PropertyTable' in smiles_data:
            smiles_property_table = smiles_data['PropertyTable']
            if 'Properties' in smiles_property_table and len(smiles_property_table['Properties']) > 0:
                canonical_smiles = smiles_property_table['Properties'][0]['CanonicalSMILES']

    # Make a request to the API for the same compounds 3D information
    try:
        # Receive the file as an SDF file
        sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d"
        sdf_response = requests.get(sdf_url)
        if sdf_response.status_code == 200:
            sdf_text = sdf_response.text
            # Parse SDF text to extract atoms, bonds, coordinates, and charge
            parsed_coords, parsed_bonds, parsed_charges = parse_sdf(sdf_text)
    except:
        return None
    return canonical_smiles, parsed_coords, parsed_bonds, parsed_charges


# Function to gather and write all compound info for given compound ID pairs
def write_compound_info(filename, output_filename):
    # Open the given file, the name given in 'filename'
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        # Retrieve a list of column names
        fieldnames = reader.fieldnames[:]

        # Find the index of the "STITCH 1" column
        stitch1_index = fieldnames.index('# STITCH 1')
        # Insert new fieldnames to the right of "STITCH 1" - to hold new 3d values
        fieldnames.insert(stitch1_index + 1, 'C1 SMILES')
        fieldnames.insert(stitch1_index + 2, 'C1 Coords')
        fieldnames.insert(stitch1_index + 3, 'C1 Bonds')
        fieldnames.insert(stitch1_index + 4, 'C1 Charges')

        # Find the index of the "STITCH 2" column
        stitch2_index = fieldnames.index('STITCH 2')
        # Insert new fieldnames to the right of "STITCH 2" - to hold new 3d values
        fieldnames.insert(stitch2_index + 1, 'C2 SMILES')
        fieldnames.insert(stitch2_index + 2, 'C2 Coords')
        fieldnames.insert(stitch2_index + 3, 'C2 Bonds')
        fieldnames.insert(stitch2_index + 4, 'C2 Charges')

        # Decide which row of the csv to start writing from
        start_row = 0

        # Write the new data to a new file, the name given in 'output_filename'
        with open(output_filename, mode='a', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            # Write the header row only if the file is empty
            if output_file.tell() == 0:
                writer.writeheader()

            last_stitch1_id = None
            last_stitch2_id = None

            # Iterate over all rows in the input file
            for index, row in enumerate(reader):
                # If the starting row is higher, don't gather data for this row
                if index < start_row:
                    continue

                # Get the first compounds ID
                stitch1_id = row['# STITCH 1']

                # Gets the compound ID of varying length, removes trailing 0's
                compound_id1 = re.search(r'\d+', stitch1_id).group()
                compound_id1 = str(int(compound_id1))

                # Check to see if the last requested ID was the same as the current requested ID
                # If it is then it will just use the value from the last time get_compound_info was called
                if stitch1_id != last_stitch1_id:
                    # If it is not then retrieve the compounds information
                    stitch1_canonical_smiles, stitch1_parsed_coords, stitch1_parsed_bonds, stitch1_parsed_charges = get_compound_info(compound_id1)
                    if stitch1_canonical_smiles is None:
                        # If no data is returned from the API then print an error
                        print("No data found for", stitch1_id)
                        continue
                    else:
                        # Set the last requested ID
                        last_stitch1_id = stitch1_id

                # Insert the information to the new columns
                row['C1 SMILES'] = stitch1_canonical_smiles
                row['C1 Coords'] = stitch1_parsed_coords
                row['C1 Bonds'] = stitch1_parsed_bonds
                row['C1 Charges'] = stitch1_parsed_charges

                # Get the first compounds ID
                stitch2_id = row['STITCH 2']

                # Gets the compound ID of varying length, removes trailing 0's
                compound_id2 = re.search(r'\d+', stitch2_id).group()
                compound_id2 = str(int(compound_id2))

                # Check to see if the last requested ID was the same as the current requested ID
                # If it is then it will just use the value from the last time get_compound_info was called
                if stitch2_id != last_stitch2_id:
                    # If it is not then retrieve the compounds information
                    stitch2_canonical_smiles, stitch2_parsed_coords, stitch2_parsed_bonds, stitch2_parsed_charges = get_compound_info(compound_id2)
                    if stitch2_canonical_smiles is None:
                        # If no data is returned from the API then print an error
                        print("No data found for", stitch2_id)
                        continue
                    else:
                        # Set the last requested ID
                        last_stitch2_id = stitch2_id

                # Insert the information to the new columns
                row['C2 SMILES'] = stitch2_canonical_smiles
                row['C2 Coords'] = stitch2_parsed_coords
                row['C2 Bonds'] = stitch2_parsed_bonds
                row['C2 Charges'] = stitch2_parsed_charges

                # Write the compound pair to the output file
                writer.writerow(row)
    print("File complete!")


# Function to calculate distance between two atoms
def calculate_distance(atom1_coords, atom2_coords):
    x1, y1, z1 = atom1_coords
    x2, y2, z2 = atom2_coords
    # Euclidian distance formula
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance


# Function to parse string and convert to list of tuples
def parse_list(str):
    # Remove brackets and parentheses
    str = str.strip("[]()")
    # Split by "), ("
    str_list = str.split("), (")
    # Use eval to convert each coordinate string to a tuple
    str_list = [eval(item) for item in str_list]
    return str_list


# Function to calculate bond lengths for all compounds pairs in cleanedData.csv
def calculate_bond_lengths(filename, output_filename):

    # Open the given file, the name given in 'filename'
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        # Retrieve a list of column names
        fieldnames = reader.fieldnames[:]

        # Find the index of the "STITCH 1" column (first compound ID column)
        stitch_index1 = fieldnames.index('# STITCH 1')
        # Insert new fieldnames for the computed lengths
        fieldnames.insert(stitch_index1 + 5, 'C1 Computed Lengths')
        fieldnames.insert(stitch_index1 + 10, 'C2 Computed Lengths')

        # Write the new data to a new file, the name given in 'output_file'
        with open(output_filename, mode='a', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            # Write the header row only if the file is empty
            if output_file.tell() == 0:
                writer.writeheader()

            # Iterate over each row in the file being read
            for index, row in enumerate(reader):

                # Get the coordinates of the compounds
                c1_atom_coords = parse_list(row['C1 Coords'])
                c2_atom_coords = parse_list(row['C2 Coords'])

                # Get the bonds between the compounds atoms
                c1_atom_bonds = parse_list(row['C1 Bonds'])
                c2_atom_bonds = parse_list(row['C2 Bonds'])

                # Calculate the first compounds bond lengths
                c1_distances = []
                # Loop over each bond in the compounds list of bonds
                for bond1 in c1_atom_bonds:
                    # Retrieve the indexes of the atoms for the given bond, retrieving their coordinates
                    atom1_index, atom2_index, _ = bond1
                    atom1_coords = c1_atom_coords[atom1_index - 1][2:]
                    atom2_coords = c1_atom_coords[atom2_index - 1][2:]

                    # Calculate their euclidian distance, and add it to the list of distances
                    distance = calculate_distance(atom1_coords, atom2_coords)
                    c1_distances.append((atom1_index, atom2_index, distance))

                # Calculate the second compounds bond lengths
                c2_distances = []
                # Loop over each bond in the compounds list of bonds
                for bond2 in c2_atom_bonds:
                    # Retrieve the indexes of the atoms for the given bond, retrieving their coordinates
                    atom1_index, atom2_index, _ = bond2
                    atom1_coords = c2_atom_coords[atom1_index - 1][2:]
                    atom2_coords = c2_atom_coords[atom2_index - 1][2:]

                    # Calculate their euclidian distance, and add it to the list of distances
                    distance = calculate_distance(atom1_coords, atom2_coords)
                    c2_distances.append((atom1_index, atom2_index, distance))

                # Insert the information to the new columns
                row['C1 Computed Lengths'] = c1_distances
                row['C2 Computed Lengths'] = c2_distances

                # Write the whole row to the new file
                writer.writerow(row)
        print("Distances calculated!")


'''
# Testing with Aspirin:
canonical_smiles, parsed_coords, parsed_bonds, parsed_charges = get_compound_info(2244)
print("Aspirin Canonical SMILES:", canonical_smiles)
print("Aspirin Atom Info:", parsed_coords)
print("Aspirin Bond Info:", parsed_bonds)
print("Paracetamol Charge Info:", parsed_charges)
print("\n")
'''

'''WARNING: The below statements will allow for data gathering and calculating. However, this file is also imported by
   gnn_3d. If you leave these statements uncommented, then gnn_3d will attempt to run them when it imports them.
   It is therefore recommended to avoid leaving them uncommented when not directly testing these functions.'''
# RUN THIS TO WRITE THE COMPOUND INFO
#write_compound_info("ChChSe-Decagon_polypharmacy/ChChSe-Decagon_polypharmacy.csv", "ChChSe-Decagon_polypharmacy/gatheredData.csv")

# ADD THE BOND LENGTHS
#calculate_bond_lengths("ChChSe-Decagon_polypharmacy/cleanedData.csv", "ChChSe-Decagon_polypharmacy/computedData.csv")
