
import openpyxl, json, statistics
wb = openpyxl.load_workbook("/data/disk1/datasets/modma/extracted/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx")
ws = wb.active
headers = [ws.cell(1,c).value for c in range(1, ws.max_column+1)]
print("Headers:", headers)
print("Rows:", ws.max_row - 1)
for r in range(2, min(5, ws.max_row+1)):
    vals = [ws.cell(r,c).value for c in range(1, ws.max_column+1)]
    print(f"Row {r}:", vals)
phq9 = [float(ws.cell(r,7).value) for r in range(2, ws.max_row+1) if ws.cell(r,7).value is not None]
print(f"PHQ-9: N={len(phq9)} mean={statistics.mean(phq9):.1f} std={statistics.stdev(phq9):.1f} min={min(phq9)} max={max(phq9)}")
print(f"PHQ-9>=10: {sum(1 for x in phq9 if x>=10)}/{len(phq9)}")
# Column 7 = PHQ-9, check what's in last column
last_col = [ws.cell(r, ws.max_column).value for r in range(2, ws.max_row+1)]
print("Last column values:", last_col[:10])
