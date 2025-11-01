import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

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

# Define fuzzy rules based on research findings
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

# ============= CSV PROCESSING FUNCTION =============

def calculate_employability_from_csv(csv_file):
    """
    Read university data from CSV and calculate employability scores
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with university rankings data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with original data plus employability scores and categories
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Map column names from CSV to fuzzy system variables
    column_mapping = {
        'ar score': 'academic_reputation',
        'er score': 'employer_reputation',
        'fsr score': 'faculty_student_ratio',
        'cpp score': 'citations_per_paper',
        'swp score': 'staff_phd'
    }
    
    # Initialize lists to store results
    employability_scores = []
    employability_categories = []
    
    # Process each university
    for idx, row in df.iterrows():
        try:
            # Extract input values
            inputs = {
                'academic_reputation': float(row['ar score']) if pd.notna(row['ar score']) else 50,
                'employer_reputation': float(row['er score']) if pd.notna(row['er score']) else 50,
                'faculty_student_ratio': float(row['fsr score']) if pd.notna(row['fsr score']) else 50,
                'citations_per_paper': float(row['cpp score']) if pd.notna(row['cpp score']) else 50,
                'staff_phd': float(row['swp score']) if pd.notna(row['swp score']) else 50
            }
            
            # Set inputs to fuzzy system
            for key, value in inputs.items():
                employability_simulation.input[key] = value
            
            # Compute employability
            employability_simulation.compute()
            score = employability_simulation.output['employability']
            
            # Categorize employability
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
            print(f"Error processing row {idx} ({row.get('Institution', 'Unknown')}): {e}")
            employability_scores.append(None)
            employability_categories.append('Error')
    
    # Add results to dataframe
    df['Employability_Score'] = employability_scores
    df['Employability_Category'] = employability_categories
    
    return df

# ============= USAGE EXAMPLE =============

# Process CSV file
csv_filename = 'dataset.csv'  # Replace with your CSV filename
results_df = calculate_employability_from_csv(csv_filename)

# Display results
print("\n" + "="*100)
print("GRADUATE EMPLOYABILITY ASSESSMENT RESULTS")
print("="*100 + "\n")

# Show key columns
display_columns = ['Institution', 'Country / Territory', 'ar score', 'er score', 
                   'fsr score', 'cpp score', 'swp score', 
                   'Employability_Score', 'Employability_Category']

print(results_df[display_columns].head(20).to_string(index=False))

# Save results to new CSV
output_filename = 'university_employability_results.csv'
results_df.to_csv(output_filename, index=False)
print(f"\n\nResults saved to: {output_filename}")

# ============= STATISTICS AND ANALYSIS =============

print("\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100 + "\n")

# Category distribution
category_counts = results_df['Employability_Category'].value_counts()
print("Employability Category Distribution:")
print(category_counts)
print()

# Score statistics
print("Employability Score Statistics:")
print(f"Mean: {results_df['Employability_Score'].mean():.2f}")
print(f"Median: {results_df['Employability_Score'].median():.2f}")
print(f"Min: {results_df['Employability_Score'].min():.2f}")
print(f"Max: {results_df['Employability_Score'].max():.2f}")
print(f"Std Dev: {results_df['Employability_Score'].std():.2f}")

# Top 10 universities by employability
print("\n" + "="*100)
print("TOP 10 UNIVERSITIES BY EMPLOYABILITY SCORE")
print("="*100 + "\n")

top_10 = results_df.nlargest(10, 'Employability_Score')[
    ['RANK', 'Institution', 'Country / Territory', 'Employability_Score', 'Employability_Category']
]
print(top_10.to_string(index=False))

# ============= OPTIONAL: SINGLE UNIVERSITY QUERY =============

def query_university_employability(df, university_name):
    """
    Query employability for a specific university
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with employability results
    university_name : str
        Name of university to query
    """
    result = df[df['Institution'].str.contains(university_name, case=False, na=False)]
    
    if len(result) > 0:
        print(f"\n{'='*80}")
        print(f"EMPLOYABILITY ASSESSMENT: {result.iloc[0]['Institution']}")
        print(f"{'='*80}\n")
        print(f"Country: {result.iloc[0]['Country / Territory']}")
        print(f"Rank: {result.iloc[0]['RANK']}")
        print(f"\nInput Scores:")
        print(f"  Academic Reputation: {result.iloc[0]['ar score']:.1f}")
        print(f"  Employer Reputation: {result.iloc[0]['er score']:.1f}")
        print(f"  Faculty-Student Ratio: {result.iloc[0]['fsr score']:.1f}")
        print(f"  Citations per Paper: {result.iloc[0]['cpp score']:.1f}")
        print(f"  Staff with PhD: {result.iloc[0]['swp score']:.1f}")
        print(f"\n{'='*80}")
        print(f"EMPLOYABILITY SCORE: {result.iloc[0]['Employability_Score']:.2f}")
        print(f"CATEGORY: {result.iloc[0]['Employability_Category']}")
        print(f"{'='*80}\n")
    else:
        print(f"University '{university_name}' not found in dataset")

# Example query
# query_university_employability(results_df, 'Peking University')
