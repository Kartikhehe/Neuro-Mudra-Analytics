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
    Modified for custom yoga mudra labels, robust counter handling, and column ignoring by Gemini AI.
    Includes temporary debug print statements.
"""

import sys
import numpy as np
import scipy.signal
import scipy.stats
import scipy.fftpack

# -----------------------------------------------------------------------------
# DATA LOADERS AND PRE-PROCESSORS
# -----------------------------------------------------------------------------

# REMOVED: The check_for_timestamps_and_convert function is removed
# and its logic is integrated directly into matrix_from_csv_file

def matrix_from_csv_file(filename, sampling_rate_hz):
	"""
	Loads a CSV file into a numpy matrix. It expects the first column to be
	time (or sample counter) and the following columns to be EEG signals.
	Returns the matrix and the header (list of column names).
	"""

	# Load the CSV data skipping the header row
	# Using delimiter=',' as confirmed by your latest error message
	csv_data = np.loadtxt(filename, delimiter=',', skiprows=1)
	full_matrix = np.array(csv_data)

	# Read the header separately
	with open(filename, 'r') as f:
		header_line = f.readline().strip()
	# Split header by comma, as confirmed by your latest error message
	header_list = [h.strip() for h in header_line.split(',')]

	if full_matrix.shape[0] == 0:
		raise ValueError(f"File {filename} contains no data rows.")

	# --- DEBUG PRINT: Raw matrix shape and first few rows ---
	print(f"DEBUG_LOAD: {filename} - Raw matrix shape: {full_matrix.shape}")
	print(f"DEBUG_LOAD: {filename} - First 5 rows of raw raw matrix:\n{full_matrix[:min(5, full_matrix.shape[0]), :]}")
	# --- END DEBUG PRINT ---

	# --- MODIFIED LOGIC FOR TIME COLUMN HANDLING ---
	# Ensure the first column is time in seconds
	if header_list[0].lower() == 'counter':
		print(f"Info: First column of '{filename}' is 'Counter'. Converting to timestamps (seconds) using sampling rate: {sampling_rate_hz} Hz.")
		full_matrix[:, 0] = full_matrix[:, 0] / sampling_rate_hz
	else:
		print(f"Info: First column of '{filename}' is '{header_list[0]}'. Assuming it's already timestamps (in seconds). No conversion applied.")
	# --- END MODIFIED LOGIC ---

	# --- DEBUG PRINT: Check for non-finite values in the time column after conversion ---
	if not np.isfinite(full_matrix[:, 0]).all():
		problem_rows = np.where(~np.isfinite(full_matrix[:, 0]))[0]
		print(f"ERROR: Non-finite values (NaN, Inf) found in time column of {filename} at indices: {problem_rows[:min(10, len(problem_rows))]} (first 10). This indicates data corruption or misinterpretation.")
	# --- END DEBUG PRINT ---

	# --- DEBUG PRINT: Time-converted matrix info ---
	print(f"DEBUG_LOAD: {filename} - First 5 rows of time-converted matrix:\n{full_matrix[:min(5, full_matrix.shape[0]), :]}")
	if full_matrix.shape[0] >= 2:
		print(f"DEBUG_LOAD: {filename} - Time range: {full_matrix[0,0]:.4f} to {full_matrix[-1,0]:.4f} seconds. Total duration: {(full_matrix[-1,0] - full_matrix[0,0]):.4f} seconds.")
	elif full_matrix.shape[0] == 1:
		print(f"DEBUG_LOAD: {filename} - Only 1 row in matrix. Time: {full_matrix[0,0]:.4f} seconds.")
	else:
		print(f"DEBUG_LOAD: {filename} - Matrix is empty after loading. This should have been caught earlier.")
	# --- END DEBUG PRINT ---

	return full_matrix, header_list


def get_time_slice(matrix, start = 0., period = 1.):
	"""
	Extracts a slice of the matrix based on time.

	Parameters:
		matrix (numpy.ndarray): 2D matrix with time in the first column (in seconds).
		start (float): Start time (in seconds) of the slice.
		period (float): Duration (in seconds) of the slice.

	Returns:
		numpy.ndarray: 2D matrix containing the data for the specified time slice.

	Author: [fcampelo]
	"""

	# Ensure matrix has enough rows for min/max operations, if not, return None
	if matrix is None or matrix.shape[0] < 2:
		# print(f"DEBUG_SLICE: Matrix passed to get_time_slice has < 2 rows ({matrix.shape[0] if matrix is not None else 'None'}). Returning None.")
		return None, 0.

	# Determine start and end sample indices
	idx_0 = np.searchsorted(matrix[:, 0], start, side='left')
	idx_1 = np.searchsorted(matrix[:, 0], start + period, side='left')

	# --- DEBUG PRINT: Indices found by searchsorted ---
	# To avoid overwhelming output, keep these commented unless deep-diving into a single file's slicing
	# print(f"DEBUG_SLICE: start={start:.4f}, end_target={start+period:.4f}")
	# print(f"DEBUG_SLICE: matrix_time_min={matrix[0,0]:.4f}, matrix_time_max={matrix[-1,0]:.4f}")
	# print(f"DEBUG_SLICE: idx_0={idx_0}, idx_1={idx_1}")
	# --- END DEBUG PRINT ---

	# Handle cases where the slice is at the very end of the data
	if idx_1 > matrix.shape[0]:
		idx_1 = matrix.shape[0]
		# print(f"DEBUG_SLICE: Adjusted idx_1 to matrix.shape[0]: {idx_1}")

	# This happens if the period is too short or start is too late for available data
	# Or if there's only one sample in the slice after searchsorted
	if idx_0 >= idx_1:
		# print(f"DEBUG_SLICE: idx_0 ({idx_0}) >= idx_1 ({idx_1}). Returning None (No valid range).")
		return None, 0.

	# Extract the slice
	current_slice = matrix[idx_0:idx_1, :]

	# If the slice has fewer than 2 rows, it's not suitable for resampling and feature extraction
	if current_slice.shape[0] < 2:
		# print(f"DEBUG_SLICE: current_slice has < 2 rows ({current_slice.shape[0]}). Returning None.")
		return None, 0.

	# Calculate actual duration of the extracted slice
	duration = current_slice[-1, 0] - current_slice[0, 0]
	# print(f"DEBUG_SLICE: current_slice.shape={current_slice.shape}, actual_duration={duration:.4f}")

	return current_slice, duration


# -----------------------------------------------------------------------------
# FEATURE EXTRACTORS
# -----------------------------------------------------------------------------

def extract_time_domain_features(signal, axis=0):
	"""
	Extracts basic time domain features from a signal.

	Parameters:
		signal (numpy.ndarray): The signal data.
		axis (int): The axis along which to compute the features (default: 0).

	Returns:
		numpy.ndarray: 1D array of feature values.

	Author: [lmanso] (refactored by [fcampelo])
	"""

	mean_val = np.mean(signal, axis=axis)
	std_val = np.std(signal, axis=axis)
	min_val = np.min(signal, axis=axis)
	max_val = np.max(signal, axis=axis)
	ptp_val = np.ptp(signal, axis=axis) # Peak-to-peak

	return np.array([mean_val, std_val, min_val, max_val, ptp_val]).flatten()


def extract_freq_domain_features(signal, sampling_rate_hz):
	"""
	Extracts basic frequency domain features (Power Spectral Density) using FFT.
	Focuses on common EEG bands: Delta, Theta, Alpha, Beta, Gamma.

	Parameters:
		signal (numpy.ndarray): The signal data (time domain).
		sampling_rate_hz (float): The sampling rate of the signal in Hz.

	Returns:
		numpy.ndarray: 1D array of power spectral densities for each band.

	Author: [fcampelo] (inspired by various sources)
	"""

	if signal.shape[0] < 2: # Need at least 2 points for FFT
		return np.zeros(5) # Return array of zeros if signal too short

	N = signal.shape[0] # Number of samples
	T = 1.0 / sampling_rate_hz # Sample spacing

	yf = scipy.fftpack.fft(signal, axis=0) # Perform FFT along the sample axis
	xf = scipy.fftpack.fftfreq(N, T)[:N//2] # Frequency bins (only positive frequencies)

	# Compute power spectral density
	# PSD = (1/N) * |FFT(signal)|^2
	psd = 2.0/N * np.abs(yf[0:N//2])**2 # Two-sided spectrum, take only first half

	# Define EEG frequency bands (typical ranges)
	bands = {
		'delta': (0.5, 4),
		'theta': (4, 8),
		'alpha': (8, 13),
		'beta': (13, 30),
		'gamma': (30, 100) # Often extends higher, but 100Hz is a common upper limit
	}

	band_psds = []
	for band_name, (low, high) in bands.items():
		idx_band = np.where((xf >= low) & (xf <= high))
		if idx_band[0].size > 0: # Check if there are any frequencies in the band
			band_power = np.sum(psd[idx_band])
		else:
			band_power = 0.0 # No frequencies in this band
		band_psds.append(band_power)

	return np.array(band_psds).flatten()


# -----------------------------------------------------------------------------
# MAIN FEATURE EXTRACTION ROUTINE
# -----------------------------------------------------------------------------

def generate_feature_vectors_from_samples(file_path, nsamples, period, state, remove_redundant, cols_to_ignore, sampling_rate_hz):
	"""
	Splits the whole signal into samples of period 'period', and extracts features
	using a sliding window (with 50% overlap). \s
	Returns an array where each row is a feature vector and the last column is the class label.
	"""

	# Initialise return matrix
	RETURN_MATRIX = None

	# Load the full data matrix
	matrix, header_list = matrix_from_csv_file(file_path, sampling_rate_hz) # Pass sampling_rate_hz here

	# Remove the ignored columns
	if cols_to_ignore != -1 and cols_to_ignore is not None:
		# Ensure cols_to_ignore is a list if it's a single index
		if not isinstance(cols_to_ignore, list):
			cols_to_ignore = [cols_to_ignore]

		# Adjust indices if they are negative (e.g., -1 for last column)
		adjusted_cols_to_ignore = []
		for idx in cols_to_ignore:
			if idx < 0:
				adjusted_cols_to_ignore.append(matrix.shape[1] + idx) # Convert negative to positive index
			else:
				adjusted_cols_to_ignore.append(idx)

		# Sort and remove duplicates for safety
		adjusted_cols_to_ignore = sorted(list(set(adjusted_cols_to_ignore)))

		# Filter out the time column (index 0) if it was accidentally included in cols_to_ignore
		# As np.delete works on the entire `s` slice, which includes the time column
		adjusted_cols_to_ignore = [idx for idx in adjusted_cols_to_ignore if idx != 0]

	# Loop through the signal with 50% overlap windows
	# The start of the next window is half the period from the current window's start.
	step = period / 2.0
	current_time = matrix[0, 0] # Start from the actual first timestamp in the file
	processed_slices_count = 0 # Debug counter for this file

	# --- DEBUG PRINT: Loop start info ---
	print(f"DEBUG_LOOP_START: Processing {file_path}. Initial current_time={current_time:.4f}s. Total matrix duration: {(matrix[-1,0] - matrix[0,0]):.4f}s")
	# --- END DEBUG PRINT ---

	while True:
		s, actual_duration = get_time_slice(matrix, start=current_time, period=period)

		if s is None or s.shape[0] < 2:
			# This is the primary exit point if no more valid slices can be found
			print(f"DEBUG_LOOP_END: Breaking loop for {file_path}. Current_time={current_time:.4f}s. s is None or s.shape[0] < 2. s_shape={s.shape if s is not None else 'None'}")
			break # No more full or sufficiently long slices

		if actual_duration == 0:
			# This means the slice had only 1 sample, so it's skipped
			print(f"DEBUG_LOOP_SKIP: Skipping slice for {file_path}. current_time={current_time:.4f}s. Only 1 sample in slice (duration 0).")
			current_time += step
			continue

		# Remove the ignored columns AFTER slicing for this window
		if cols_to_ignore != -1 and cols_to_ignore is not None and len(adjusted_cols_to_ignore) > 0:
			s = np.delete(s, adjusted_cols_to_ignore, axis=1) # Apply deletion on the time slice


		# Ensure the time column is removed for resampling of signal data
		# And feature extraction is applied to signal data only
		if s.shape[1] < 2: # Only time column left or no columns after deletion
			print(f"Warning: No signal channels left in slice from {current_time}s after ignoring columns. Skipping.")
			current_time += step
			continue


		# Resample each channel individually to nsamples
		# s[:, 1:] correctly selects all signal channels (excluding time column s[:,0])
		if s.shape[1] - 1 <= 0: # Ensure there's at least one signal channel to process
			print(f"Warning: No signal channels to process in slice from {current_time}s after column selection. Skipping.")
			current_time += step
			continue

		resampled_signals = np.zeros((nsamples, s.shape[1] - 1))
		resampling_failed_for_slice = False

		for i in range(s.shape[1] - 1): # Iterate over signal channels (excluding time column s[:,0])
			try:
				# Use t=s[:,0] for accurate time axis for resampling
				resampled_signal, _ = scipy.signal.resample(s[:, 1+i], num=nsamples, t=s[:,0], axis=0)
				resampled_signals[:, i] = resampled_signal
			except ValueError as e:
				print(f"Error during resampling of channel {i+1} at time {current_time}s: {e}. Skipping this slice.")
				resampling_failed_for_slice = True
				break

		if resampling_failed_for_slice: # If resampling failed for any channel
			current_time += step
			continue


		# --- Feature Extraction ---
		# Extract features from each resampled signal
		# Features will be flattened for each channel and then concatenated
		feature_vector = np.array([])
		for channel_idx in range(resampled_signals.shape[1]):
			signal_data = resampled_signals[:, channel_idx]

			# Time domain features
			td_features = extract_time_domain_features(signal_data)

			# Frequency domain features
			fd_features = extract_freq_domain_features(signal_data, sampling_rate_hz)

			feature_vector = np.concatenate((feature_vector, td_features, fd_features))

		# Add the state (class label) as the last element of the feature vector
		feature_vector = np.append(feature_vector, state)

		# Append to the return matrix
		if RETURN_MATRIX is None:
			RETURN_MATRIX = feature_vector.reshape(1, -1) # Reshape to a 2D array with 1 row
		else:
			RETURN_MATRIX = np.vstack([RETURN_MATRIX, feature_vector])

		# --- DEBUG PRINT: Success message for processed slice ---
		# This will print for every successful feature vector generation
		# print(f"DEBUG_LOOP_SUCCESS: Processed slice for {file_path} at {current_time:.4f}s. Feature vector shape: {feature_vector.shape}")
		# --- END DEBUG PRINT ---
		processed_slices_count += 1

		# Move to the next window
		current_time += step

	# If no data was processed, RETURN_MATRIX might still be None
	if RETURN_MATRIX is None:
		print(f"Warning: No valid feature vectors could be generated from {file_path}. Returning empty matrix.")
		# Return an empty array to signify no data and a placeholder header for consistency
		return np.empty((0, 1)), ['NoFeatures']

	# Generate header for the output CSV
	# Features per channel: 5 time domain (mean, std, min, max, ptp) + 5 frequency domain (band powers) = 10 features
	feature_names_per_channel = ['mean', 'std', 'min', 'max', 'ptp',
								 'delta_psd', 'theta_psd', 'alpha_psd', 'beta_psd', 'gamma_psd']

	# Get the names of the channels that were *not* ignored
	original_channel_names = [h for i, h in enumerate(header_list) if i != 0 and i not in adjusted_cols_to_ignore]

	header = []
	for ch_name in original_channel_names:
		for feat_name in feature_names_per_channel:
			header.append(f"{ch_name}_{feat_name}")
	header.append("Class") # Add the class label column

	# --- DEBUG PRINT: Summary of processed slices for the file ---
	print(f"DEBUG_LOOP_SUMMARY: Finished processing {file_path}. Total processed slices added: {processed_slices_count}")
	# --- END DEBUG PRINT ---

	return RETURN_MATRIX, header
