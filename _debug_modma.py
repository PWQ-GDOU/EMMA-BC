import pandas as pd
from pathlib import Path

xlsx_path = '/data/disk1/datasets/modma/extracted/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx'
audio_dir = Path('/data/disk1/datasets/modma/extracted/audio_lanzhou_2015')

# Load labels
df = pd.read_excel(xlsx_path)
print('XLSX columns:', df.columns.tolist())
print('XLSX first 3 rows:')
print(df.head(3))
print()

# Check subject_id column
sid_col = df.columns[0]
print(f'Subject ID column: {sid_col}')
sids_xlsx = set(str(s) for s in df[sid_col].values)
print(f'XLSX subjects: {len(sids_xlsx)}')
print(f'Sample IDs: {list(sids_xlsx)[:5]}')
print()

# Check directory subjects
dir_sids = set(d.name for d in audio_dir.iterdir() if d.is_dir())
print(f'Directory subjects: {len(dir_sids)}')
print(f'Sample dirs: {list(dir_sids)[:5]}')
print()

# Find mismatch
common = sids_xlsx & dir_sids
xlsx_only = sids_xlsx - dir_sids
dir_only = dir_sids - sids_xlsx
print(f'Common: {len(common)}')
print(f'XLSX only: {len(xlsx_only)} {list(xlsx_only)[:5] if xlsx_only else "none"}')
print(f'Dir only: {len(dir_only)} {list(dir_only)[:5] if dir_only else "none"}')
