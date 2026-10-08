# Data

The dataset (`viral-db.csv`, ~60 MB) is too large for GitHub and is hosted on Google Drive.

**Automatic:**

```bash
python scripts/download_data.py
```

**Manual:** download it from
[Google Drive](https://drive.google.com/file/d/15x7NVNCivXGwS9OpuJfa4hm0FnP3lm4c/view?usp=sharing)
and save it as `data/viral-db.csv`.

| Property | Value |
|---|---|
| Rows | 72,244 |
| Unique threads (`topic_id`) | 1,195 |
| Forums (`forum_id`) | 2 |
| Columns | 70 (2 IDs, 8 virality-definition parameters, 59 features, `class_label`) |
| Features used | 25 (`rf*` and `ef*` families) |
| Label | `class_label` - 1 = viral (43%), 0 = not viral (57%) |

Each thread appears many times under different virality-definition parameters
(`alpha`, `beta`, `sigma_sus`, `sigma_fos`, ...), which is why splitting is done by
`topic_id` rather than by row.
