import pandas as pd

def ADtarget(drug_file, ad_genes_file, output_file):
    xls = pd.ExcelFile(drug_file) # provide drug target file
    sheet_names = xls.sheet_names[:9] 

    df_ad_genes = pd.read_csv(ad_genes_file) # provide AD target genes
    ad_genes = set(df_ad_genes['hgnc_symbol'].unique())

    overlap_results = []

    for sheet_name in sheet_names:
        df_drugs = pd.read_excel(xls, sheet_name=sheet_name)
        grouped = df_drugs.groupby('Drug Name')
        for drug_name, group in grouped:
            drug_genes = set(group['Gene Name'].unique())
            overlap_genes = drug_genes.intersection(ad_genes) # collect AD hit genes for each drug

            # store results
            if overlap_genes:
                for gene in overlap_genes:
                    overlap_results.append({
                        'Drug Name': drug_name,
                        'Gene Name': gene
                    })
            else:
                overlap_results.append({
                    'Drug Name': drug_name,
                    'Gene Name': "doesn't hit AD target"
                })

    overlap_df = pd.DataFrame(overlap_results)
    overlap_df.to_excel(output_file, index=False)

    return overlap_df

def complementary_exposure_baseline(overlap_df, combination_df):
    combination_results = []

    for _, row in combination_df.iterrows():
        # extract drugs from the combination
        drug_A = row['Drug A']
        drug_B = row['Drug B']
        
        if drug_A in overlap_df['Drug Name'].values: # drug A has gene information
            overlapA_genes = set(overlap_df[(overlap_df['Drug Name'] == drug_A) & (overlap_df['Gene Name'] != "doesn't hit AD target")]['Gene Name'])
            len_overlapA = len(overlapA_genes) if overlapA_genes else 0 
        else:
            overlapA_genes = set()
            len_overlapA = 'No Information' 

        if drug_B in overlap_df['Drug Name'].values:
            overlapB_genes = set(overlap_df[(overlap_df['Drug Name'] == drug_B) & (overlap_df['Gene Name'] != "doesn't hit AD target")]['Gene Name'])
            len_overlapB = len(overlapB_genes) if overlapB_genes else 0  
        else:
            overlapB_genes = set()
            len_overlapB = 'No Information'  
        
        # Calculate union and intersection for overlaps
        if len_overlapA != 'No Information' and len_overlapB != 'No Information':
            union_overlap = overlapA_genes.union(overlapB_genes)
            len_union_overlap = len(union_overlap)
            if len_overlapA == 0 or len_overlapB == 0:
                is_overlap = True
            else:
                is_overlap = not overlapA_genes.isdisjoint(overlapB_genes)
        else:
            len_union_overlap = 'NA'
            is_overlap = 'NA'
        
        if is_overlap == 'NA':
            efficacy = 'NA'
        elif is_overlap:
            efficacy = 'Non-positive'
        else:
            efficacy = 'Positive'
        
        combination_results.append({
            'Drug A': drug_A,
            'Drug B': drug_B,
            'len(overlapA)': len_overlapA,
            'len(overlapB)': len_overlapB,
            'len(union_overlap)': len_union_overlap,
            'is_overlap': is_overlap,
            'Efficacy': efficacy
        })

    res_df = pd.DataFrame(combination_results)
    
    return res_df

