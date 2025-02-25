import pandas as pd

def convert_xlsx_to_csv(input_file, output_file):
    # Load the Excel file
    df = pd.read_excel(input_file, dtype=str)
    
    # Save as CSV with UTF-8 encoding
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'CSV file saved as: {output_file}')

# Usage
convert_xlsx_to_csv('sms_data.xlsx', 'sms_data.csv')
