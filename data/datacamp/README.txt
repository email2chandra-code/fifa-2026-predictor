DataCamp competition fixture files (local copies)
=================================================

DataLab often does not offer a download button for data/ files.
These CSVs are generated from the FIFA official schedule
(data/2026_FIFA_World_Cup_Schedule.csv.csv) and formatted for competition submission.

Regenerate:
  python scripts/build_datacamp_fixtures.py

Verify against DataLab (paste in a notebook cell):
  import pandas as pd
  gf = pd.read_csv("data/group_fixtures.csv")
  print(gf.shape, gf.columns.tolist())
  print(gf.head(3).to_string())

Compare match_id and team names with your local copies in this folder.

Competition notebook (submission reference):
  - Group matches: match_id 1-72
  - Knockout matches: match_id 73-104 (32 slots)
  - Multipliers: Group ×1, R32 ×1, R16 ×2, QF ×4, SF ×8, 3P ×8, Final ×16

Known gaps vs local schedule copy:
  - Local group_fixtures: 70 rows (Group I missing 2 in source schedule)
  - Local knockout starts at match_id 71, not 73 — use DataLab files for submission match_id alignment
  - DataCamp/DataLab fixture files define submission format only; FIFA.com is the schedule authority
