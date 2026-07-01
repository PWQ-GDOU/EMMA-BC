
import openpyxl, statistics, json
wb = openpyxl.load_workbook("/data/disk1/datasets/modma/extracted/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx")
ws = wb.active

# Build proper data
data = []
for r in range(2, ws.max_row + 1):
    sid = str(ws.cell(r, 1).value)
    # Skip description rows
    if not sid or not sid[0].isdigit():
        continue
    row = {
        "subject_id": sid,
        "type": str(ws.cell(r, 2).value),
        "age": ws.cell(r, 3).value,
        "gender": str(ws.cell(r, 4).value),
        "education": ws.cell(r, 5).value,
        "PHQ_9": ws.cell(r, 6).value,
        "CTQ_SF": ws.cell(r, 7).value,
        "LES": ws.cell(r, 8).value,
        "SSRS": ws.cell(r, 9).value,
        "GAD_7": ws.cell(r, 10).value,
        "PSQI": ws.cell(r, 11).value,
    }
    data.append(row)

print(f"Valid subjects: {len(data)}")
print(f"MDD: {sum(1 for d in data if d['type']=='MDD')}, HC: {sum(1 for d in data if d['type']=='HC')}")

# PHQ-9 stats
phq9 = [d["PHQ_9"] for d in data if d["PHQ_9"] is not None]
print(f"PHQ-9: N={len(phq9)} mean={statistics.mean(phq9):.1f} std={statistics.stdev(phq9):.1f} min={min(phq9)} max={max(phq9)}")
print(f"PHQ-9>=10: {sum(1 for x in phq9 if x>=10)}/{len(phq9)}")

# All scale stats
for scale in ["PHQ_9", "GAD_7", "PSQI", "CTQ_SF", "LES", "SSRS"]:
    vals = [d[scale] for d in data if d[scale] is not None]
    if vals:
        print(f"{scale}: mean={statistics.mean(vals):.1f} std={statistics.stdev(vals):.1f} range=[{min(vals)},{max(vals)}]")

# Show first 3 subjects
for d in data[:3]:
    print(d)

# PHQ-9 histogram
bins = [(0,5),(5,10),(10,15),(15,20),(20,25),(25,30)]
for lo,hi in bins:
    cnt = sum(1 for x in phq9 if lo <= x < hi)
    bar = '#' * cnt
    print(f"  PHQ-9 [{lo:2d},{hi:2d}): {cnt:2d} {bar}")
