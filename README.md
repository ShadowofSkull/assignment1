# Graduate Employability Assessment System
## Fuzzy Inference System for SDG 4: Quality Education

A decision-support system using fuzzy logic to assess graduate employability from university quality metrics, aligned with UN Sustainable Development Goal 4 (Quality Education).

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Input Data Format](#input-data-format)
- [Output](#output)
- [Code Documentation](#code-documentation)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a **functional fuzzy inference system** to evaluate graduate employability based on university quality indicators from QS Asia University Rankings. The system uses five key attributes that research has shown to be strong predictors of graduate employability:

1. **Academic Reputation** - Institutional quality perception
2. **Employer Reputation** - Direct employer assessment of graduates  
3. **Faculty-Student Ratio** - Quality of education delivery
4. **Citations per Paper** - Research intensity indicator
5. **Staff with PhD** - Faculty qualification level

The fuzzy model categorizes universities into four employability levels: **Poor**, **Average**, **Good**, and **Excellent**.

---

## Features

- ✅ Processes QS Asia Rankings CSV data with multi-level headers
- ✅ Implements Mamdani-type fuzzy inference system with 8 expert rules
- ✅ Handles missing data gracefully with default values
- ✅ Generates employability scores (0-100) and categorical ratings
- ✅ Provides statistical analysis and top university rankings
- ✅ Exports results to CSV for further analysis
- ✅ Filters only relevant attributes for SDG 4 assessment

---

## Prerequisites

- **Python 3.7+** (Python 3.8 or higher recommended)
- **pip** package manager
- **Virtual environment** support (venv)

---


## Installation

### Step 1: Clone or Download the Project

If using git
```shell
git clone <your-repository-url>
cd graduate-employability-assessment
```
Or download and extract the ZIP file, then navigate to the folder
```shell
cd graduate-employability-assessment
```


### Step 2: Create Virtual Environment

Create an isolated Python environment for this project:

**On Windows:**
```shell
python -m venv venv
```


**On macOS/Linux:**
```shell
python3 -m venv venv
```


### Step 3: Activate Virtual Environment

**On Windows:**
```shell
venv\Scripts\activate
```


**On macOS/Linux:**
```shell
source venv/bin/activate
```


You should see `(venv)` prefix in your terminal prompt indicating the virtual environment is active.

### Step 4: Install Dependencies

Install all required packages from `requirements.txt`:
```shell
pip install -r requirements.txt
```

### Step 5: Verify Installation

Check that all packages are installed correctly:
```shell
pip list
```


You should see packages like: `numpy`, `pandas`, `scikit-fuzzy`, `matplotlib`

---

## Usage

### Basic Usage

1. **Prepare your data file**: Place your dataset CSV file in the project directory
2. **Run the analysis**:

python employability_assessment.py



### Custom Data File

To use a different CSV file, modify the filename in the script:

In employability_assessment.py, change this line:
filename = 'your-custom-file.csv'



### Expected Output

The program will:
1. Load the CSV file
2. Extract the 5 key attributes
3. Calculate employability scores using fuzzy inference
4. Display top 20 universities by employability score
5. Show summary statistics
6. Save results to `QS_Asia_2025_Employability_Results.csv`

---

## Input Data Format

The system expects a QS Asia Rankings CSV file with the following columns:

| Column Name | Data Type |
|--------------|-----------|
| `Institution Name` | String |
| `Country/ Territory` | String |
| `2025` | Integer |
| `Academic Reputation` | Float |
| `Employer Reputation` | Float |
| `Faculty Student` | Float |
| `Citations per Paper` | Float |
| `Staff with PhD` | Float |


**Note**: The CSV may have multi-level headers. The code automatically handles this structure.

---

## Output

### Console Output

The program displays:
- Number of universities processed
- Top 20 universities by employability score
- Category distribution (Excellent/Good/Average/Poor)
- Statistical summary (mean, median, std, min, max)

### CSV Output

**File**: `QS_Asia_2025_Employability_Results.csv`

Contains columns:
- `Rank` - QS ranking
- `Institution` - University name
- `Country` - Country/territory
- `Academic_Reputation` - Input score
- `Employer_Reputation` - Input score
- `Faculty_Student_Ratio` - Input score
- `Citations_per_Paper` - Input score
- `Staff_with_PhD` - Input score
- `Employability_Score` - Computed fuzzy output (0-100)
- `Employability_Category` - Classification (Excellent/Good/Average/Poor)

---

# Code Documentation
## Graduate Employability Fuzzy Inference System

---

## System Architecture

This fuzzy inference system implements a **Mamdani-type fuzzy reasoning approach** to transform university quality metrics into graduate employability assessments. The system follows a structured pipeline from data ingestion through fuzzy inference to actionable output.

Input Data → Data Processing → Fuzzification → Rule Evaluation →
Aggregation → Defuzzification → Employability Score → Classification


---

## Core Components

### 1. Fuzzy Inference Engine

**Purpose**: Transforms crisp input values into fuzzy employability assessments using linguistic rules

**Architecture**: Mamdani-type FIS with 5 inputs, 1 output, and 8 expert rules.

**Input Universe**: All inputs operate on [0, 100] scale

**Output Universe**: Employability score on [0, 100] scale

**Linguistic Variables**:
- **Low**: Membership peak at 0-25
- **Medium**: Membership peak at 25-75  
- **High**: Membership peak at 75-100

**Defuzzification Method**: Centroid (center of gravity)

---

### 2. Knowledge Base (Rule Set)

The system encodes expert knowledge through 8 fuzzy IF-THEN rules that capture relationships between university quality indicators and graduate employability.

**Rule Categories**:

**Excellence Rules** (2 rules):
- High reputation indicators → Excellent employability
- Combined high academic quality → Excellent employability

**Quality Rules** (3 rules):
- Strong employer-faculty relationships → Good employability
- Research excellence with reputation → Good employability
- Quality teaching with qualified staff → Good employability

**Baseline Rules** (2 rules):
- Medium indicators → Average employability
- Low core factors → Poor employability

**Risk Rules** (1 rule):
- Deficiency in key factors → Average employability (safety net)

**Rule Interaction**: Uses fuzzy AND (minimum) and OR (maximum) operators for compound conditions.

---

### 3. Data Processing Pipeline

**Function**: `read_qs_asia_rankings(filename)`

**Responsibility**: Extract and normalize university ranking data for fuzzy inference

**Processing Stages**:

1. **Raw Data Ingestion**
   - Reads CSV with multi-level headers
   - Handles latin-1 encoding for international characters
   - Manages malformed rows gracefully

2. **Column Mapping**
   - Maps complex QS column names to system variables
   - Extracts only 5 relevant attributes (filtering 32 unused columns)
   - Preserves metadata (rank, institution, country)

3. **Data Cleaning**
   - Removes header rows disguised as data (e.g., "Rank display")
   - Validates numeric rank values
   - Filters universities from non-university entries

4. **Type Conversion**
   - Converts score strings to float type
   - Handles missing values (NaN)
   - Validates data ranges

**Output**: Clean DataFrame with 8 columns ready for fuzzy inference

---

### 4. Inference Engine

**Function**: `calculate_employability_scores(df)`

**Responsibility**: Apply fuzzy inference to each university record

**Inference Process** (per university)

1. **Input Validation**
   - Check for missing values
   - Apply default value (50) for missing scores
   - Ensure values in valid range [0, 100]

2. **Fuzzification**
   - Convert crisp inputs to membership degrees
   - Each input mapped to Low/Medium/High fuzzy sets
   - Example: score 75 → {Low: 0, Medium: 0.5, High: 0.5}

3. **Rule Firing**
   - Evaluate all 8 rules with current inputs
   - Calculate rule strength using fuzzy operators
   - Example: IF (AR=high AND ER=high) fires with strength = min(0.8, 0.9) = 0.8

4. **Aggregation**
   - Combine fired rule outputs using MAX operator
   - Creates aggregated fuzzy output set

5. **Defuzzification**
   - Apply centroid method to get crisp score
   - Output: Single employability value [0, 100]

6. **Classification**
   - Apply threshold-based categorization
   - Thresholds: <40 Poor, 40-60 Average, 60-80 Good, ≥80 Excellent

**Robustness**: Error handling captures and logs processing failures without halting batch processing

---

## Configuration & Tuning

### Membership Function Parameters

**Current Configuration**: Triangular membership functions with 50% overlap




### Key Design Decisions

1. **Attribute Selection**: Based on empirical research showing Academic Reputation and Employer Reputation as strongest predictors of employability

2. **Default Values**: Missing scores default to 50 (medium) to avoid bias

3. **Encoding**: Uses `latin-1` encoding to handle special characters in university names

4. **Row Filtering**: Automatically removes header rows by checking for numeric rank values

5. **Output Structure**: Minimal columns (10 vs 37) for clarity and SDG 4 alignment

---

## Methodology

### Theoretical Foundation

This system aligns with **SDG 4 (Quality Education)**, specifically:
- **Target 4.4**: Increasing skills for employment and entrepreneurship
- **Target 4.3**: Equal access to quality education

### Fuzzy Inference Process

1. **Fuzzification**: Convert crisp input values (e.g., 85.5) to fuzzy sets (e.g., 0.7 high, 0.3 medium)

2. **Rule Evaluation**: Apply all 8 rules and determine firing strength

3. **Aggregation**: Combine outputs from all rules

4. **Defuzzification**: Convert fuzzy output to crisp employability score using centroid method

### Research Basis

The 5 selected attributes were chosen based on:
- Research on Southeast Asian universities (notable association with employability)
- Exclusion of student mobility metrics (weak correlation)
- Focus on institutional quality and research intensity
- Alignment with SDG 4 quality indicators

---

## Project Structure
```
graduate-employability-assessment/
│
├── employability_assessment.py # Main script
├── requirements.txt # Python dependencies
├── README.md # This file
│
├── 2025-QS-Asia-Rankings-Results-v1.5-For-qs.com-2-_3.csv # Input data
├── QS_Asia_2025_Employability_Results.csv # Output (generated)
│
└── venv/ # Virtual environment (not committed to git)
```


---