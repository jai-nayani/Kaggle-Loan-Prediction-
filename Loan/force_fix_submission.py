import pandas as pd
import csv

# Read predictions
preds = pd.read_csv('Loan/submission_safe.csv')
pred_map = dict(zip(preds['id'], preds['loan_paid_back']))

# Read sample submission and replace values line by line
output_rows = []
with open('Loan/sample_submission.csv', 'r') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    
    for row in reader:
        row_id = int(row['id'])
        if row_id in pred_map:
            row['loan_paid_back'] = f"{pred_map[row_id]:.6f}" # Format to 6 decimals
        else:
            print(f"Warning: ID {row_id} missing from predictions")
            
        output_rows.append(row)

# Write with forced standard formatting
with open('Loan/submission_final_fix.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

print("Created Loan/submission_final_fix.csv")

