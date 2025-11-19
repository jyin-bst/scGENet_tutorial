#!/usr/bin/env python3

import pandas as pd
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process gene cluster data.')
parser.add_argument('-i', '--input', required=True, help='Input txt file with gene and cluster information')
parser.add_argument('-o', '--output', required=True, help='Output txt file for the binary matrix')

args = parser.parse_args()

# Read the input file (assuming tab-delimited)
data = pd.read_csv(args.input, sep='\t', index_col=0)

# Extract the unique clusters
unique_clusters = data['cluster'].unique()

# Create a new DataFrame with binary matrix
binary_matrix = pd.DataFrame(0, index=data.index, columns=unique_clusters)

# Assign 1 to the corresponding cluster column for each gene
for gene, cluster in data['cluster'].items():
    binary_matrix.loc[gene, cluster] = 1

# Write the binary matrix to the output file
binary_matrix.to_csv(args.output, sep='\t')

print(f'Binary matrix written to {args.output}')

