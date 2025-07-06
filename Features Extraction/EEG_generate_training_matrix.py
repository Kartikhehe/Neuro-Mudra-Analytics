#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
## Version history:

2018:
	Original script by Dr. Luis Manso [lmanso], Aston University

2019, June:
	Revised, commented and updated by Dr. Felipe Campelo [fcampelo], Aston University
	(f.campelo@aston.ac.uk / fcampelo@gmail.com)

2024, June:
    Modified for custom yoga mudra labels and specific column ignoring by Gemini AI.
"""

import os, sys
import numpy as np
import pandas as pd # Import pandas to read CSV headers
# Ensure EEG_feature_extraction.py is in the same directory or accessible in PYTHONPATH
from EEG_feature_extraction import generate_feature_vectors_from_samples


# --- IMPORTANT: SET YOUR EEG DEVICE'S ACTUAL SAMPLING RATE HERE ---
# This is crucial for correctly converting 'Counter' columns to time in seconds.
ACTUAL_EEG_SAMPLING_RATE_HZ = 256.0 # <--- CHANGE THIS IF YOUR DEVICE IS DIFFERENT (e.g., 128.0, 500.0, etc.)


def gen_training_matrix(directory_path, output_file, cols_to_ignore):
	"""
	Reads the csv files in directory_path and assembles the training matrix with
	the features extracted using the functions from EEG_feature_extraction.

	Parameters:
		directory_path (str): directory containing the CSV files to process.
		output_file (str): filename for the output file.
		cols_to_ignore (list): list of column indices to ignore from the CSV
                                (indices refer to the columns in the original CSV file, 0-based).

 	Returns:
		numpy.ndarray: 2D matrix containing the data read from the CSV

	Author:
		Original: [lmanso]
		Updates and documentation: [fcampelo]
        Modified: Gemini AI
	"""

	# Initialise return matrix
	FINAL_MATRIX = None

	for x in os.listdir(directory_path):

		# Ignore non-CSV files
		if not x.lower().endswith('.csv'):
			continue

		# For safety we'll ignore files containing the substring "test".
		# [Test files should not be in the dataset directory in the first first place]
		if 'test' in x.lower():
			continue

        # --- MODIFICATION START: Custom Mudra Label Parsing ---
        # Assuming your filename format is 'name-mudraname-suffix.csv' or 'name-mudraname.csv'.
        # Example: 'participantA-Jnana-session1.csv' -> mudra_name = 'jnana'
		try:
			# x[:-4] removes '.csv' from the filename
			parts = x[:-4].split('-')
			if len(parts) >= 2:
				name = parts[0] # This part 'name' is not directly used for labeling, but extracted
				mudra_name = parts[1].lower() # Convert to lowercase for consistent dictionary lookup
			else:
				raise ValueError(
                    f"Filename '{x}' does not match expected format: 'yourname-mudraname-anyidentifier.csv' or 'yourname-mudraname.csv'"
                )
		except Exception as e:
			print (f'Error parsing file name "{x}": {e}')
			print ('Please ensure filename format is "yourname-mudraname-anyidentifier.csv" or "yourname-mudraname.csv"')
			sys.exit(-1)

        # Define your mudra-to-label mapping here.
        # IMPORTANT:
        # 1. Ensure these keys exactly match the lowercase mudra names you use in your filenames.
        # 2. Assign unique numerical labels (floats, as the original script uses for states).
        # You can change these numerical values as long as they are distinct for each mudra.
		mudra_labels = {
           'chandra': 0.0,
             'kamal': 1.0,
             'kanista': 2.0,
             'mrigi': 3.0,
             'mushti': 4.0,
             'prana': 5.0,
             'pranam': 6.0,
             'samana': 7.0,
             'vajra': 8.0,
             'yoni': 9.0,
           
            # Make sure ALL your mudra names from your filenames are listed here
            # with unique numerical labels!
		}

		if mudra_name in mudra_labels:
			state = mudra_labels[mudra_name]
		else:
			print (f'Unknown yoga mudra state "{mudra_name}" found in file name "{x}"')
			print ('Please ensure the mudra name in the filename matches one of the predefined mudras in the `mudra_labels` dictionary.')
			sys.exit(-1)
        # --- MODIFICATION END: Custom Mudra Label Parsing ---

		print (f'Using file "{x}" (Assigned Label: {mudra_name} -> {int(state)})') # Added informative print
		full_file_path = os.path.join(directory_path, x) # Use os.path.join for better path handling

		vectors, header = generate_feature_vectors_from_samples(
            file_path = full_file_path,
            nsamples = 150,  # Keeping at 150 - your files are large enough
            period = 1.,     # Keeping at 1.0 second period
            state = state,
            remove_redundant = True,
            cols_to_ignore = cols_to_ignore,
            sampling_rate_hz = ACTUAL_EEG_SAMPLING_RATE_HZ # <--- Pass the defined sampling rate
        )

        # Handle case where no vectors were generated for a file
		if vectors.shape[0] == 0:
			print(f"Skipping file {x} as no valid feature vectors were generated.")
			continue # Move to the next file

		print ('Resulting vector shape for the file:', vectors.shape)


		if FINAL_MATRIX is None:
			FINAL_MATRIX = vectors
		else:
			FINAL_MATRIX = np.vstack( [ FINAL_MATRIX, vectors ] )

	if FINAL_MATRIX is None:
		print("Error: No feature vectors were generated from any files. Check your input directory, filenames, and data quality.")
		sys.exit(-1)


	print ('FINAL_MATRIX shape after processing all files:', FINAL_MATRIX.shape)

	# Shuffle rows
	np.random.shuffle(FINAL_MATRIX)
	print('FINAL_MATRIX rows shuffled.')

	# Save to file
	np.savetxt(output_file, FINAL_MATRIX, delimiter = ',',
			header = ','.join(header),
			comments = '')
	print(f'Extracted features saved to: {output_file}')

	return None


if __name__ == '__main__':
	"""
	Main function. The parameters for the script are the following:
		[1] directory_path: The directory where the script will look for the files to process.
		[2] output_file: The filename of the generated output file.

	Author:
		Original by [lmanso]
		Documentation: [fcampelo]
        Modified: Gemini AI
"""
	if len(sys.argv) < 3:
		print ('\nUsage: python EEG_generate_training_matrix.py <input_directory_path> <output_file_name.csv>')
		print ('Example: python EEG_generate_training_matrix.py ./my_eeg_data_folder extracted_features.csv\n')
		sys.exit(-1)

	directory_path = sys.argv[1]
	output_file = sys.argv[2]

    # --- NEW LOGIC: Identify columns to ignore by name ---
    # Find the first valid CSV file to read its header
	first_csv_file = None
	for fname in os.listdir(directory_path):
		if fname.lower().endswith('.csv') and 'test' not in fname.lower():
			first_csv_file = os.path.join(directory_path, fname)
			break

	if first_csv_file is None:
		print("Error: No valid CSV files found in the input directory. Please check the path and file names.")
		sys.exit(-1)

	cols_to_ignore_indices = []
	try:
        # Read only the header row
		df_header = pd.read_csv(first_csv_file, nrows=0)
		all_columns = df_header.columns.tolist()

		# Define the column names you want to ignore
        # IMPORTANT: Ensure these names exactly match your CSV header names (case-sensitive)
		columns_to_exclude_by_name = []

		for col_name_to_ignore in columns_to_exclude_by_name:
			if col_name_to_ignore in all_columns:
                # Get the 0-based index of the column
				cols_to_ignore_indices.append(all_columns.index(col_name_to_ignore))
			else:
				print(f"Warning: Column '{col_name_to_ignore}' not found in header of '{first_csv_file}'. It will not be ignored.")

        # Sort and remove duplicates for robustness, though not strictly necessary for np.delete
		cols_to_ignore_indices = sorted(list(set(cols_to_ignore_indices)))

	except Exception as e:
		print(f"Error reading header from '{first_csv_file}' to determine columns to ignore: {e}")
		sys.exit(-1)

	print(f"Columns to ignore (by index in original CSV): {cols_to_ignore_indices}")
    # --- END NEW LOGIC ---

    # Pass the dynamically determined columns to ignore
	gen_training_matrix(directory_path, output_file, cols_to_ignore = cols_to_ignore_indices)

	
