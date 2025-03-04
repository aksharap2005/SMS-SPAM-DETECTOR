import os

file_path = r"c:\Users\PC\OneDrive\Documents\demo\sms_spam.cvs.xlsx"

if os.path.exists(file_path):
    print("File found, proceeding to load...")
    import pandas as pd
    data = pd.read_csv(file_path, encoding='latin-1')
else:
    print("File not found! Check the path:", file_path)
