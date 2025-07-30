#!/bin/bash
# Download datasets for DyHuCoG experiments

DATASET=$1
DATA_DIR="data"

# Create data directory if it doesn't exist
mkdir -p $DATA_DIR

# Function to download and extract MovieLens-100K
download_ml100k() {
    echo "Downloading MovieLens-100K dataset..."
    cd $DATA_DIR
    
    if [ -d "ml-100k" ]; then
        echo "MovieLens-100K already exists. Skipping download."
        return
    fi
    
    wget -q --show-progress http://files.grouplens.org/datasets/movielens/ml-100k.zip
    unzip -q ml-100k.zip
    rm ml-100k.zip
    
    echo "MovieLens-100K downloaded successfully!"
    cd ..
}

# Function to download and extract MovieLens-1M
download_ml1m() {
    echo "Downloading MovieLens-1M dataset..."
    cd $DATA_DIR
    
    if [ -d "ml-1m" ]; then
        echo "MovieLens-1M already exists. Skipping download."
        return
    fi
    
    wget -q --show-progress http://files.grouplens.org/datasets/movielens/ml-1m.zip
    unzip -q ml-1m.zip
    rm ml-1m.zip
    
    echo "MovieLens-1M downloaded successfully!"
    cd ..
}

# Function to download Amazon Book dataset
download_amazon_book() {
    echo "Downloading Amazon Book dataset..."
    cd $DATA_DIR
    
    if [ -d "amazon-book" ]; then
        echo "Amazon Book dataset already exists. Skipping download."
        return
    fi
    
    mkdir -p amazon-book
    cd amazon-book
    
    # Download from the source (update URL as needed)
    echo "Note: Amazon Book dataset requires manual download from:"
    echo "http://jmcauley.ucsd.edu/data/amazon/"
    echo "Please download 'Books' category and extract to data/amazon-book/"
    
    cd ../..
}

# Main logic
case $DATASET in
    "ml-100k")
        download_ml100k
        ;;
    "ml-1m")
        download_ml1m
        ;;
    "amazon-book")
        download_amazon_book
        ;;
    "all")
        download_ml100k
        download_ml1m
        download_amazon_book
        ;;
    *)
        echo "Usage: $0 {ml-100k|ml-1m|amazon-book|all}"
        echo "Example: $0 ml-100k"
        exit 1
        ;;
esac

echo "Dataset download completed!"