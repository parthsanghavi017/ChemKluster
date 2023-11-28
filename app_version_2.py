import streamlit as st
import pandas as pd
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
from rdkit.Chem import PandasTools
from sklearn.cluster import KMeans
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import base64
import os
from zipfile import ZipFile

# Set Streamlit app configuration
st.set_page_config(page_title="Molecule Clustering App V2")
st.header("Molecule Clustering App")


# Introduction to the app
st.write("""
This app allows you to upload a CSV file containing molecule data with valid SMILES strings. You can then perform K-means clustering on the molecules and visualize the clusters along with their Scaffold MCS.

Please make sure your CSV file has the following format:
- The CSV should have at least two columns: 'ID' and 'SMILES'.
- 'ID' column should contain molecule identifiers.
- 'SMILES' column should contain valid SMILES strings representing the molecules.

Demo File link (Download as CSV):
[Download Demo CSV File](https://drive.google.com/uc?export=download&id=1-NyLsUneGRuLpjCS0LzxvDcQPlnkuyxh)

After uploading the CSV file and specifying the range of K values for clustering, click the 'Run' button to perform clustering and visualize the results for each K value.
""")
st.subheader("By Afroz and Parth")

# Function to highlight molecules
def highlight_molecules(molecules, molecule_names, mcs, number, label=True, same_orientation=True, **kwargs):
    molecules = deepcopy(molecules)
    # convert MCS to molecule
    pattern = Chem.MolFromSmarts(mcs.smartsString)
    # find the matching atoms in each molecule
    matching = [molecule.GetSubstructMatch(pattern) for molecule in molecules[:number]]

    legends = None
    if label is True:
        legends = molecule_names[:number]

    # Align by matched substructure so they are depicted in the same orientation
    # Adapted from: https://gist.github.com/greglandrum/82d9a86acb3b00d3bb1df502779a5810
    if same_orientation:
        mol, match = molecules[0], matching[0]
        AllChem.Compute2DCoords(mol)
        coords = [mol.GetConformer().GetAtomPosition(x) for x in match]
        coords2D = [Geometry.Point2D(pt.x, pt.y) for pt in coords]
        for mol, match in zip(molecules[1:number], matching[1:number]):
            if not match:
                continue
            coord_dict = {match[i]: coord for i, coord in enumerate(coords2D)}
            AllChem.Compute2DCoords(mol, coordMap=coord_dict)

    return Draw.MolsToGridImage(
        molecules[:number],
        legends=legends,
        molsPerRow=5,
        highlightAtomLists=matching
    )

# Perform k-means clustering and visualization
def perform_clustering(mol_df, mols, k_range):
    best_mcs_scaffolds = []
    mcs_strings_per_k = []
    cluster_data_per_k = []
    cluster_counts_per_k = []

    for k in k_range:
        # Convert molecules to fingerprints
        fps = [Chem.RDKFingerprint(mol) for mol in mols]

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(fps)

        # Extract MCS for each cluster
        mcs_clusters = []
        for cluster_id in range(k):
            cluster_molecules = [mols[i] for i, cluster_label in enumerate(clusters) if cluster_label == cluster_id]
            mcs = rdFMCS.FindMCS(cluster_molecules)
            mcs_clusters.append(mcs)

        # Find the best MCS scaffold for this K value
        best_mcs = max(mcs_clusters, key=lambda mcs: mcs.numAtoms * mcs.numBonds)
        best_mcs_scaffolds.append(best_mcs.smartsString)

        # Collect the MCS strings for each cluster in a list
        mcs_strings = [mcs.smartsString for mcs in mcs_clusters]
        mcs_strings_per_k.append(mcs_strings)

        # Store cluster data in a DataFrame
        cluster_df = pd.DataFrame({"Molecule": mol_df["Molecule"], "SMILES": mol_df["SMILES"], "Cluster": clusters})
        cluster_data_per_k.append(cluster_df)

        # Calculate number of molecules in each cluster
        cluster_counts = cluster_df["Cluster"].value_counts().sort_index()
        cluster_counts_per_k.append(cluster_counts)

        # Visualize clusters and MCS
        for cluster_id, mcs in enumerate(mcs_clusters):
            st.subheader(f"Cluster {cluster_id + 1} - Scaffold MCS:")
            st.write("MCS SMARTS string:", mcs.smartsString)

            # Get the molecule names and SMILES for the current cluster
            cluster_molecules = cluster_df[cluster_df["Cluster"] == cluster_id]["Molecule"].tolist()
            cluster_smiles = cluster_df[cluster_df["Cluster"] == cluster_id]["SMILES"].tolist()

            # Get the indices of the molecules in the original DataFrame
            molecule_indices = mol_df.index[mol_df["Molecule"].isin(cluster_molecules)].tolist()

            # Get the corresponding molecules with the correct names
            cluster_mols = [mols[idx] for idx in molecule_indices]

            # Draw substructure from Smarts
            mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
            st.image(Draw.MolToImage(mcs_mol), caption=f"Cluster {cluster_id + 1} - Scaffold")

            # Highlight MCS for each molecule in the cluster
            img = highlight_molecules(cluster_mols, cluster_molecules, mcs, number=len(cluster_mols), label=True, useSVG=True)
            st.image(img)

        # Add a separator between clusters of different K values
        if k != k_range[-1]:
            st.markdown("---")  # Add a horizontal separator

    return best_mcs_scaffolds, mcs_strings_per_k, cluster_data_per_k, cluster_counts_per_k



# Main function to run the Streamlit app
def main():
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        mol_df = pd.read_csv(uploaded_file)

        # Only keep molecules with valid SMILES
        mol_df = mol_df.dropna(subset=['SMILES'])

        # Add molecule column to data frame
        PandasTools.AddMoleculeColumnToFrame(mol_df, "SMILES")

        # Set the molecule name column (replace "Molecule" with the correct column name)
        mol_df.rename(columns={"ID": "Molecule"}, inplace=True)

        # Create a list to hold the molecules
        mols = mol_df["ROMol"].tolist()

        st.write(f"Set with {len(mols)} molecules loaded.")

        # Set range of K values for clustering
        k_min = st.number_input("Minimum K value", min_value=2, max_value=100, value=2)
        k_max = st.number_input("Maximum K value", min_value=2, max_value=100, value=10)

        # Display the "Run" button
        if st.button("Run"):
            k_range = range(k_min, k_max + 1)

            # Perform clustering
            best_mcs_scaffolds, mcs_strings_per_k, cluster_data_per_k, cluster_counts_per_k = perform_clustering(mol_df, mols, k_range)

            # Display clustering results
            progress_bar = st.progress(0)
            progress_status = st.empty()

            for i, (k, cluster_df, mcs_strings, cluster_counts) in enumerate(zip(k_range, cluster_data_per_k, mcs_strings_per_k, cluster_counts_per_k)):
                progress_status.text(f"Processing results for K={k}...")
                time.sleep(1)  # Simulate processing time
                progress_bar.progress((i + 1) / len(k_range))

               # Export cluster data to CSV
                cluster_df_with_clusters = cluster_df.copy()  # Make a copy of the DataFrame to avoid modifying the original one
                cluster_df_with_clusters["Cluster"] = cluster_df_with_clusters["Cluster"] + 1  # Increment cluster values to start from 1

                # Merge cluster data with original data
                final_cluster_data = pd.merge(mol_df, cluster_df_with_clusters, on="Molecule")

                # Select columns to keep in the final CSV
                columns_to_keep = final_cluster_data.columns.difference(["ROMol", "SMILES_y"])
                final_cluster_data = final_cluster_data[columns_to_keep]

                csv_file = final_cluster_data.to_csv(index=False)  # Use the updated DataFrame
                b64 = base64.b64encode(csv_file.encode()).decode()  # Convert the CSV file to bytes
                file_name = f"cluster_data_k{k}.csv"
                href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download cluster data (K={k}).csv</a>'
                st.markdown(href, unsafe_allow_html=True)



                # Save scaffold and cluster images, and generate cluster assignment CSV
                folder_name = f"cluster_k{k}_files"
                os.makedirs(folder_name, exist_ok=True)

            

                # Create a zip file for each K value containing all relevant files
                with ZipFile(f"{folder_name}.zip", "w") as zipf:
                    for file_path in [f"{folder_name}/{file}" for file in os.listdir(folder_name)]:
                        zipf.write(file_path, os.path.basename(file_path))


                # Display cluster counts
                st.subheader(f"Cluster Counts (K={k})")
                # Increment the index of cluster_counts by 1 to start from 1
                cluster_counts.index = cluster_counts.index + 1
                st.bar_chart(cluster_counts)


            progress_status.text("Processing completed!")

    else:
        st.write("Please upload a CSV file.")


if __name__ == "__main__":
    main()
