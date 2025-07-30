# Data Directory

This directory contains the datasets used for DyHuCoG experiments.

## Supported Datasets

### MovieLens-100K
- **Download**: Run `bash scripts/download_data.sh ml-100k`
- **Files**:
  - `u.data`: Ratings (user, item, rating, timestamp)
  - `u.item`: Item information with genres
  - `u.user`: User demographics
  - `u.genre`: Genre list

### MovieLens-1M
- **Download**: Run `bash scripts/download_data.sh ml-1m`
- **Files**:
  - `ratings.dat`: Ratings
  - `movies.dat`: Movie information
  - `users.dat`: User information

### Amazon Book (Optional)
- **Download**: Manual download required from [Amazon dataset](http://jmcauley.ucsd.edu/data/amazon/)
- Place files in `data/amazon-book/`

## Directory Structure

```
data/
├── ml-100k/          # MovieLens-100K dataset
│   ├── u.data
│   ├── u.item
│   ├── u.user
│   └── ...
├── ml-1m/            # MovieLens-1M dataset
│   ├── ratings.dat
│   ├── movies.dat
│   └── users.dat
├── amazon-book/      # Amazon Book dataset
│   └── ratings.csv
└── processed/        # Preprocessed data
    ├── ml-100k/
    │   ├── train.csv
    │   ├── val.csv
    │   ├── test.csv
    │   └── stats.csv
    └── ...
```

## Preprocessing

To preprocess the raw data:

```bash
python scripts/preprocess.py --dataset ml-100k --min_interactions 5
```

This will:
1. Filter users and items with minimum interactions
2. Convert to implicit feedback (ratings > 3)
3. Create temporal train/val/test splits
4. Save processed files in `data/processed/`

## Data Statistics

After preprocessing, statistics are saved in `stats.csv`:
- Number of users
- Number of items  
- Number of interactions
- Data density
- Train/val/test split sizes

## Custom Datasets

To use your own dataset:

1. Format your data with columns: `user`, `item`, `rating`, `timestamp`
2. Place in `data/your-dataset/`
3. Modify `src/data/dataset.py` to add a loader
4. Run preprocessing script

## Notes

- All user and item IDs are reindexed to be consecutive starting from 1
- Timestamps are used for temporal splitting
- Implicit feedback conversion uses threshold of 3.0 by default