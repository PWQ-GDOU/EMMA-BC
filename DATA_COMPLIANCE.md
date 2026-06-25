# Data Compliance Notice

DAIC-WOZ and MODMA datasets contain sensitive clinical health data.

## Obligations
1. Store data in **access-controlled directories** only
2. **Never commit** raw data, transcripts, labels, or audio to public repositories
3. Follow your institution's **IRB/ethics committee** guidelines
4. **Anonymize** all published results (no participant IDs, no raw transcripts)
5. Delete data when no longer needed per your **Data Use Agreement**

## Dataset Access
| Dataset  | Source | License | Path |
|----------|--------|---------|------|
| DAIC-WOZ | USC    | Required | /data/disk1/datasets/diac_woz/ |
| MODMA    | LZU    | Required | /data/disk1/datasets/modma/ |
| RAVDESS  | Zenodo | CC BY 4.0 | /data/disk1/datasets/ravdess/ |

## Configuration
All data paths are configurable — **nothing is hardcoded**: see constructor arguments in `phaseB/multimodal_dataset.py`
