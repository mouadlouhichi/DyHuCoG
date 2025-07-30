#!/bin/bash
# Download datasets for DyHuCoG experiments - Enhanced for Q2 paper requirements

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
    
    # Download preprocessed version from RecBole
    echo "Downloading preprocessed Amazon Book dataset..."
    wget -q --show-progress https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Amazon/Amazon_Books.zip
    unzip -q Amazon_Books.zip
    rm Amazon_Books.zip
    
    # Rename files to expected format
    if [ -f "Amazon_Books.inter" ]; then
        mv Amazon_Books.inter train.txt
    fi
    
    echo "Amazon Book dataset downloaded successfully!"
    cd ../..
}

# Function to download Yelp 2018 dataset
download_yelp2018() {
    echo "Downloading Yelp 2018 dataset..."
    cd $DATA_DIR
    
    if [ -d "yelp2018" ]; then
        echo "Yelp 2018 dataset already exists. Skipping download."
        return
    fi
    
    mkdir -p yelp2018
    cd yelp2018
    
    # Download from Google Drive (preprocessed version)
    echo "Downloading preprocessed Yelp 2018 dataset..."
    
    # Using gdown to download from Google Drive
    pip install -q gdown
    
    # Yelp 2018 preprocessed files
    gdown --quiet "https://drive.google.com/uc?id=1_L2tnVeDqHm_lYJmPcq5sxgqTJre9MqC" -O yelp2018_train.txt
    gdown --quiet "https://drive.google.com/uc?id=1_QqzJI-U9bHJGTnwPuFUzhTX2N5AHPAM" -O yelp2018_test.txt
    
    # Rename to expected format
    mv yelp2018_train.txt train.txt
    mv yelp2018_test.txt test.txt
    
    # Create business categories file (placeholder)
    echo "{}" > business_categories.json
    
    echo "Yelp 2018 dataset downloaded successfully!"
    cd ../..
}

# Function to download Gowalla dataset
download_gowalla() {
    echo "Downloading Gowalla dataset..."
    cd $DATA_DIR
    
    if [ -d "gowalla" ]; then
        echo "Gowalla dataset already exists. Skipping download."
        return
    fi
    
    mkdir -p gowalla
    cd gowalla
    
    # Download from SNAP Stanford
    echo "Downloading Gowalla check-in dataset..."
    wget -q --show-progress https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz
    gunzip loc-gowalla_totalCheckins.txt.gz
    
    # Process into train/test format
    echo "Processing Gowalla dataset..."
    python3 ../../scripts/process_gowalla.py
    
    echo "Gowalla dataset downloaded and processed successfully!"
    cd ../..
}

# Function to download Amazon Electronics dataset
download_amazon_electronics() {
    echo "Downloading Amazon Electronics dataset..."
    cd $DATA_DIR
    
    if [ -d "amazon-electronics" ]; then
        echo "Amazon Electronics dataset already exists. Skipping download."
        return
    fi
    
    mkdir -p amazon-electronics
    cd amazon-electronics
    
    # Download from UCSD repository
    echo "Downloading Amazon Electronics ratings..."
    wget -q --show-progress http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv
    
    # Download metadata (for categories)
    echo "Downloading Amazon Electronics metadata..."
    wget -q --show-progress http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
    gunzip meta_Electronics.json.gz
    
    echo "Amazon Electronics dataset downloaded successfully!"
    cd ../..
}

# Function to download preprocessed datasets from RecBole
download_recbole_datasets() {
    echo "Downloading additional datasets from RecBole..."
    cd $DATA_DIR
    
    # Create RecBole directory
    mkdir -p recbole_datasets
    cd recbole_datasets
    
    # Base URL for RecBole datasets
    RECBOLE_BASE="https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets"
    
    # Download multiple datasets
    datasets=("Amazon_Beauty" "Amazon_Sports_and_Outdoors" "Yelp2022")
    
    for dataset in "${datasets[@]}"; do
        if [ ! -d "$dataset" ]; then
            echo "Downloading $dataset..."
            wget -q --show-progress "$RECBOLE_BASE/Amazon/${dataset}.zip" || \
            wget -q --show-progress "$RECBOLE_BASE/Yelp/${dataset}.zip" || \
            echo "Failed to download $dataset"
            
            if [ -f "${dataset}.zip" ]; then
                unzip -q "${dataset}.zip"
                rm "${dataset}.zip"
                echo "$dataset downloaded successfully!"
            fi
        fi
    done
    
    cd ../..
}


# Function to verify all downloads
verify_downloads() {
    echo ""
    echo "Verifying downloaded datasets..."
    echo "================================"
    
    datasets=("ml-100k" "ml-1m" "yelp2018" "gowalla" "amazon-electronics" "amazon-book")
    
    for dataset in "${datasets[@]}"; do
        if [ -d "$DATA_DIR/$dataset" ]; then
            echo "✓ $dataset - Downloaded"
        else
            echo "✗ $dataset - Missing"
        fi
    done
    
    echo ""
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
    "yelp2018"|"yelp")
        download_yelp2018
        ;;
    "gowalla")
        download_gowalla
        ;;
    "amazon-electronics")
        download_amazon_electronics
        ;;
    "recbole")
        download_recbole_datasets
        ;;
    "all")
        echo "Downloading all datasets for Q2 paper evaluation..."
        download_ml100k
        download_ml1m
        download_yelp2018
        download_gowalla
        download_amazon_electronics
        download_amazon_book
        verify_downloads
        ;;
    "q2-minimal")
        echo "Downloading minimal Q2 dataset collection (4 datasets)..."
        download_ml100k
        download_ml1m
        download_yelp2018
        download_gowalla
        verify_downloads
        ;;
    *)
        echo "Enhanced Dataset Download Script for DyHuCoG Q2 Paper"
        echo "===================================================="
        echo ""
        echo "Usage: $0 {dataset-name|all|q2-minimal}"
        echo ""
        echo "Individual datasets:"
        echo "  ml-100k           - MovieLens 100K (movie ratings)"
        echo "  ml-1m             - MovieLens 1M (movie ratings)"
        echo "  yelp2018          - Yelp 2018 (business reviews)"
        echo "  gowalla           - Gowalla (location check-ins)"
        echo "  amazon-electronics - Amazon Electronics (product ratings)"
        echo "  amazon-book       - Amazon Books (book ratings)"
        echo "  recbole           - Additional RecBole datasets"
        echo ""
        echo "Collections:"
        echo "  all               - Download all datasets"
        echo "  q2-minimal        - Download 4 core datasets for Q2 paper"
        echo ""
        echo "Example: $0 ml-100k"
        echo "Example: $0 q2-minimal"
        exit 1
        ;;
esac

echo "Dataset download completed!"