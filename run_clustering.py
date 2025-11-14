"""
Script to run clustering analysis on existing time series data.
This script loads precomputed time series data and runs clustering analysis.
"""

from kbc_cleaned import load_time_series_data, analyze_and_cluster

def main():
    # Load the precomputed time series data
    print("Loading time series data...")
    _, _, time_series_data = load_time_series_data()
    
    if time_series_data is None:
        print("Error: No time series data found!")
        return
    
    print("\nRunning clustering analysis...")
    analyze_and_cluster(
        time_series_data=time_series_data,
        selected_k=[4, 6, 8],  # k values for clustering
        regenerate_dtw=True,   # Force recompute distance matrices
        regenerate_clustering=True,  # Force recompute clustering
        demographic_file="data.csv"
    )
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main() 