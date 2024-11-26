from baseline_functions import ADtarget, complementary_exposure_baseline
import pandas as pd

drug_file = 'List of AD Drugs & Target Genes.xlsx'
ad_genes_file = 'Agora-gene-list.csv'
output_file = 'AD_hits_results.xlsx'

overlap_df = ADtarget(drug_file, ad_genes_file, output_file)
print("Overlap results saved to:", output_file)

combination_file = 'drug combination for baseline model.xlsx'
combination_df = pd.read_excel(combination_file)

# Run complementary_exposure_baseline function
result_df = complementary_exposure_baseline(overlap_df, combination_df)
result_df.to_excel('baseline_result.xlsx', index=False)
