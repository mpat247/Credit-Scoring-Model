import csv
import os

# Define mappings for categorical attributes
attr_mappings = {
    1: { 
        'A11': '... < 0 DM', 
        'A12': '0 <= ... < 200 DM', 
        'A13': '... >= 200 DM / salary assignments for at least 1 year', 
        'A14': 'no checking account'
    },
    3: { 
        'A30': 'no credits taken / all credits paid back duly', 
        'A31': 'all credits at this bank paid back duly', 
        'A32': 'existing credits paid back duly till now', 
        'A33': 'delay in paying off in the past', 
        'A34': 'critical account / other credits existing (not at this bank)'
    },
    4: { 
        'A40': 'car (new)', 
        'A41': 'car (used)', 
        'A42': 'furniture/equipment', 
        'A43': 'radio/television', 
        'A44': 'domestic appliances', 
        'A45': 'repairs', 
        'A46': 'education', 
        'A47': '(vacation - does not exist?)', 
        'A48': 'retraining', 
        'A49': 'business', 
        'A410': 'others'
    },
    6: { 
        'A61': '... < 100 DM', 
        'A62': '100 <= ... < 500 DM', 
        'A63': '500 <= ... < 1000 DM', 
        'A64': '.. >= 1000 DM', 
        'A65': 'unknown/ no savings account'
    },
    7: { 
        'A71': 'unemployed', 
        'A72': '... < 1 year', 
        'A73': '1 <= ... < 4 years', 
        'A74': '4 <= ... < 7 years', 
        'A75': '.. >= 7 years'
    },
    9: { 
        'A91': 'male : divorced/separated', 
        'A92': 'female : divorced/separated/married', 
        'A93': 'male : single', 
        'A94': 'male : married/widowed', 
        'A95': 'female : single'
    },
    10: { 
        'A101': 'none', 
        'A102': 'co-applicant', 
        'A103': 'guarantor'
    },
    12: { 
        'A121': 'real estate', 
        'A122': 'building society savings agreement/ life insurance', 
        'A123': 'car or other, not in attribute 6', 
        'A124': 'unknown / no property'
    },
    14: { 
        'A141': 'bank', 
        'A142': 'stores', 
        'A143': 'none'
    },
    15: { 
        'A151': 'rent', 
        'A152': 'own', 
        'A153': 'for free'
    },
    17: { 
        'A171': 'unemployed/ unskilled - non-resident', 
        'A172': 'unskilled - resident', 
        'A173': 'skilled employee / official', 
        'A174': 'management/ self-employed / highly qualified employee/ officer'
    },
    19: { 
        'A191': 'none', 
        'A192': 'yes, registered under the customer\'s name'
    },
    20: { 
        'A201': 'yes', 
        'A202': 'no'
    }
}

# Define headers for the CSV
headers = [
    'Status of existing checking account',
    'Duration in month',
    'Credit history',
    'Purpose',
    'Credit amount',
    'Savings account/bonds',
    'Present employment since',
    'Installment rate in percentage of disposable income',
    'Personal status and sex',
    'Other debtors / guarantors',
    'Present residence since',
    'Property',
    'Age in years',
    'Other installment plans',
    'Housing',
    'Number of existing credits at this bank',
    'Job',
    'Number of people being liable to provide maintenance for',
    'Telephone',
    'Foreign worker',
    'Class'
]

# Class label mapping
class_mapping = {
    '1': 'Good',
    '2': 'Bad'
}

# Define indices of numerical attributes for type conversion
# Attributes 2,5,8,11,13,16,18 are numerical based on descriptions
numerical_attributes = [2, 5, 8, 11, 13, 16, 18]

def process_german_data(input_file, output_file):
    processed_data = []
    duplicate_checker = set()
    total_records = 0
    skipped_records = 0
    duplicate_records = 0

    with open(input_file, 'r') as infile:
        for line_number, line in enumerate(infile, start=1):
            tokens = line.strip().split()
            
            # Expecting 21 tokens: 20 attributes + 1 class label
            if len(tokens) != 21:
                print(f"Warning: Line {line_number} does not have 21 tokens. Skipping.")
                skipped_records += 1
                continue

            # Check for duplicate records
            record_key = tuple(tokens)
            if record_key in duplicate_checker:
                print(f"Info: Duplicate record found at line {line_number}. Skipping.")
                duplicate_records += 1
                continue
            duplicate_checker.add(record_key)
            total_records += 1

            record = []
            skip_record = False  # Flag to skip records with invalid data

            for i in range(20):
                token = tokens[i]
                attr_index = i + 1  # Attributes are 1-based

                if attr_index in attr_mappings:
                    mapped_value = attr_mappings[attr_index].get(token, None)
                    if mapped_value is None:
                        print(f"Warning: Unrecognized code '{token}' for attribute {attr_index} at line {line_number}. Skipping record.")
                        skip_record = True
                        break
                    # Handle 'unknown' as None or keep as-is
                    if 'unknown' in mapped_value.lower():
                        mapped_value = None  # You can choose to keep it as 'unknown' or set to None
                    record.append(mapped_value)
                else:
                    # Numerical attribute: convert to integer
                    try:
                        num = int(token)
                        record.append(num)
                    except ValueError:
                        print(f"Warning: Invalid numerical value '{token}' for attribute {attr_index} at line {line_number}. Skipping record.")
                        skip_record = True
                        break

            if skip_record:
                skipped_records += 1
                continue

            # Process class label
            class_label = tokens[20]
            mapped_class = class_mapping.get(class_label, None)
            if mapped_class is None:
                print(f"Warning: Unrecognized class label '{class_label}' at line {line_number}. Skipping record.")
                skipped_records += 1
                continue
            record.append(mapped_class)

            processed_data.append(record)

    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(processed_data)

    print(f"Processing complete.")
    print(f"Total records processed: {total_records}")
    print(f"Records skipped due to errors: {skipped_records}")
    print(f"Duplicate records skipped: {duplicate_records}")
    print(f"Clean CSV file saved as '{output_file}' with {len(processed_data)} records.")

if __name__ == "__main__":
    # Define the input and output file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_filename = 'german.data'
    output_filename = 'german.csv'
    input_path = os.path.join(current_dir, input_filename)
    output_path = os.path.join(current_dir, output_filename)

    # Check if the input file exists
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_filename}' does not exist in the directory '{current_dir}'.")
    else:
        process_german_data(input_path, output_path)
