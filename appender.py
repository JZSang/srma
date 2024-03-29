# Function that appends two csv files knowing that their columns are the exact same using pandas
import pandas as pd
def append_csv(file1, file2, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = pd.concat([df1, df2])
    df.to_csv(output_file, index=False)
    print("Files appended successfully")
    return True

append_csv('reinfection_review_211650_excluded_csv_20240303130126.csv', 'reinfection_review_211650_irrelevant_csv_20240303130132.csv', 'reinfection_review_211650_irrelevant_excluded_csv_20240303130132.csv')