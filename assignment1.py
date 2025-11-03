import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ============= FUZZY SYSTEM SETUP =============

# Define input variables (5 core attributes for employability)
academic_reputation = ctrl.Antecedent(np.arange(0, 101, 1), 'academic_reputation')
employer_reputation = ctrl.Antecedent(np.arange(0, 101, 1), 'employer_reputation')
faculty_student_ratio = ctrl.Antecedent(np.arange(0, 101, 1), 'faculty_student_ratio')
citations_per_paper = ctrl.Antecedent(np.arange(0, 101, 1), 'citations_per_paper')
staff_phd = ctrl.Antecedent(np.arange(0, 101, 1), 'staff_phd')

# Define output variable
employability = ctrl.Consequent(np.arange(0, 101, 1), 'employability')

# Define membership functions for inputs
for variable in [academic_reputation, employer_reputation, faculty_student_ratio,
                 citations_per_paper, staff_phd]:
    variable['low'] = fuzz.trimf(variable.universe, [0, 0, 50])
    variable['medium'] = fuzz.trimf(variable.universe, [0, 50, 100])
    variable['high'] = fuzz.trimf(variable.universe, [50, 100, 100])

# Define membership functions for output
employability['poor'] = fuzz.trimf(employability.universe, [0, 0, 40])
employability['average'] = fuzz.trimf(employability.universe, [20, 50, 80])
employability['good'] = fuzz.trimf(employability.universe, [60, 80, 100])
employability['excellent'] = fuzz.trimf(employability.universe, [80, 100, 100])

# Define fuzzy rules
rule1 = ctrl.Rule(academic_reputation['high'] & employer_reputation['high'],
                  employability['excellent'])
rule2 = ctrl.Rule(employer_reputation['high'] & staff_phd['high'],
                  employability['good'])
rule3 = ctrl.Rule(citations_per_paper['high'] & academic_reputation['high'],
                  employability['good'])
rule4 = ctrl.Rule(faculty_student_ratio['high'] & staff_phd['high'],
                  employability['good'])
rule5 = ctrl.Rule(academic_reputation['low'] & employer_reputation['low'],
                  employability['poor'])
rule6 = ctrl.Rule(employer_reputation['medium'] & citations_per_paper['medium'],
                  employability['average'])
rule7 = ctrl.Rule(academic_reputation['high'] & citations_per_paper['high'] &
                  staff_phd['high'], employability['excellent'])
rule8 = ctrl.Rule(faculty_student_ratio['low'] | staff_phd['low'],
                  employability['average'])

# Create control system
employability_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5,
                                         rule6, rule7, rule8])
employability_simulation = ctrl.ControlSystemSimulation(employability_ctrl)

# ============= READ QS ASIA RANKINGS FILE =============

def read_qs_asia_rankings(filename):
    """
    Read the QS Asia Rankings CSV file and extract only relevant columns
    """
    # Read with multi-level headers and latin-1 encoding
    df = pd.read_csv(filename, encoding='latin-1', header=[0, 1])

    # Flatten column names
    df.columns = [' '.join(col).strip() for col in df.columns.values]

    # Skip the first row which contains "Sorting", "RANK", etc. (header labels)
    df = df.iloc[1:].reset_index(drop=True)

    # Define columns to keep
    columns_to_keep = {
        'Unnamed: 1_level_0 2025': 'Rank',
        'Unnamed: 3_level_0 Institution Name': 'Institution',
        'Unnamed: 4_level_0 Country/ Territory': 'Country',
        'Unnamed: 11_level_0 Academic Reputation': 'Academic_Reputation',
        'Unnamed: 13_level_0 Employer Reputation': 'Employer_Reputation',
        'Unnamed: 15_level_0 Faculty Student': 'Faculty_Student_Ratio',
        'Unnamed: 19_level_0 Citations per Paper': 'Citations_per_Paper',
        'Unnamed: 23_level_0 Staff with PhD': 'Staff_with_PhD'
    }

    # Select only the columns we need
    df_selected = df[list(columns_to_keep.keys())].copy()

    # Rename columns to simpler names
    df_selected = df_selected.rename(columns=columns_to_keep)

    # Remove rows where Rank contains non-numeric values (like "Rank display")
    # This filters out any remaining header rows
    df_selected = df_selected[pd.to_numeric(df_selected['Rank'], errors='coerce').notna()].copy()

    # Convert Rank to integer
    df_selected['Rank'] = df_selected['Rank'].astype(int)

    # Convert score columns to numeric
    score_columns = ['Academic_Reputation', 'Employer_Reputation', 'Faculty_Student_Ratio',
                     'Citations_per_Paper', 'Staff_with_PhD']
    for col in score_columns:
        df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')

    # Reset index after filtering
    df_selected = df_selected.reset_index(drop=True)

    return df_selected

# ============= CALCULATE EMPLOYABILITY =============

def calculate_employability_scores(df):
    """
    Calculate employability scores for all universities
    """
    employability_scores = []
    employability_categories = []

    print(f"Processing {len(df)} universities...")

    for idx, row in df.iterrows():
        try:
            # Extract input values (use 50 as default for missing values)
            inputs = {
                'academic_reputation': float(row['Academic_Reputation']) if pd.notna(row['Academic_Reputation']) else 50,
                'employer_reputation': float(row['Employer_Reputation']) if pd.notna(row['Employer_Reputation']) else 50,
                'faculty_student_ratio': float(row['Faculty_Student_Ratio']) if pd.notna(row['Faculty_Student_Ratio']) else 50,
                'citations_per_paper': float(row['Citations_per_Paper']) if pd.notna(row['Citations_per_Paper']) else 50,
                'staff_phd': float(row['Staff_with_PhD']) if pd.notna(row['Staff_with_PhD']) else 50
            }

            # Set inputs to fuzzy system
            for key, value in inputs.items():
                employability_simulation.input[key] = value

            # Compute employability
            employability_simulation.compute()
            score = employability_simulation.output['employability']

            # Categorize
            if score >= 80:
                category = 'Excellent'
            elif score >= 60:
                category = 'Good'
            elif score >= 40:
                category = 'Average'
            else:
                category = 'Poor'

            employability_scores.append(round(score, 2))
            employability_categories.append(category)

        except Exception as e:
            employability_scores.append(None)
            employability_categories.append('Error')

    df['Employability_Score'] = employability_scores
    df['Employability_Category'] = employability_categories

    return df

# ============= MAIN EXECUTION =============

print("="*100)
print("FUZZY INFERENCE SYSTEM FOR SDG 4: GRADUATE EMPLOYABILITY ASSESSMENT")
print("QS Asia University Rankings 2025")
print("="*100 + "\n")

# Read the file
filename = 'dataset.csv'
print(f"Reading file: {filename}")
df = read_qs_asia_rankings(filename)
print(f"✓ Successfully loaded {len(df)} universities")
print(f"✓ Removed header rows (Sorting, Rank display, etc.)")
print(f"✓ Extracted 5 key attributes for employability assessment\n")

# Calculate employability
df = calculate_employability_scores(df)
print(f"✓ Calculated employability scores\n")

# Display all columns (only the used attributes)
print("="*100)
print("ATTRIBUTES USED IN THE MODEL:")
print("="*100)
print("1. Academic Reputation")
print("2. Employer Reputation")
print("3. Faculty Student Ratio")
print("4. Citations per Paper")
print("5. Staff with PhD")
print("\n" + "="*100)
print("TOP 20 UNIVERSITIES BY EMPLOYABILITY SCORE")
print("="*100 + "\n")

valid_results = df[df['Employability_Category'] != 'Error']
top_20 = valid_results.nlargest(20, 'Employability_Score')
print(top_20.to_string(index=False))

# Summary Statistics
print("\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100 + "\n")

category_counts = valid_results['Employability_Category'].value_counts()
print("Employability Category Distribution:")
for category, count in category_counts.items():
    percentage = (count / len(valid_results)) * 100
    print(f"  {category:12s}: {count:4d} ({percentage:5.1f}%)")

print(f"\nEmployability Score Statistics:")
print(f"  Mean:   {valid_results['Employability_Score'].mean():.2f}")
print(f"  Median: {valid_results['Employability_Score'].median():.2f}")
print(f"  Std:    {valid_results['Employability_Score'].std():.2f}")
print(f"  Min:    {valid_results['Employability_Score'].min():.2f}")
print(f"  Max:    {valid_results['Employability_Score'].max():.2f}")

# Save results with only relevant columns
output_filename = 'QS_Asia_2025_Employability_Results.csv'
df.to_csv(output_filename, index=False, encoding='utf-8')
print(f"\n✓ Results saved to: {output_filename}")
print(f"✓ Output contains {len(df)} universities with {len(df.columns)} relevant columns")

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
