"""
Hair Tag Filtering Script - Frequency ≥ 400
Filters to keep only tags with 400+ occurrences and removes images without these tags
"""

import pandas as pd
import json
import os
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_CSV = "/Users/apple/Desktop/Fall_2025/DS5660-DL/mini_project/data_fetch/labels_combined.csv"              # UPDATE: Your original CSV
FREQUENCY_CSV = "/Users/apple/Desktop/Fall_2025/DS5660-DL/mini_project/data_fetch/all_hair_tags_frequency.csv" # The CSV from tag_analysis.py
OUTPUT_DIR = "/Users/apple/Desktop/Fall_2025/DS5660-DL/mini_project/data_fetch/filtered_data"                   # Where to save outputs
THRESHOLD = 400                                # Minimum frequency

# ============================================================================
# STEP 1: Load and Filter Tags by Frequency
# ============================================================================

def load_frequent_tags(frequency_csv_path, threshold):
    """
    Load tags from all_hair_tags_frequency.csv and filter by threshold.
    
    Args:
        frequency_csv_path: Path to all_hair_tags_frequency.csv
        threshold: Minimum frequency (e.g., 400)
    
    Returns:
        List of tags that meet the threshold
    """
    print("="*70)
    print(f"STEP 1: Loading Tags with Frequency ≥ {threshold}")
    print("="*70)
    
    # Load the frequency CSV
    freq_df = pd.read_csv(frequency_csv_path)
    print(f"\nLoaded {len(freq_df)} hair tags from {frequency_csv_path}")
    
    # Filter by threshold
    filtered_df = freq_df[freq_df['count'] >= threshold]
    kept_tags = filtered_df['tag'].tolist()
    
    print(f"\n✓ Kept {len(kept_tags)} tags with frequency ≥ {threshold}")
    print(f"\nTags kept:")
    for i, row in filtered_df.iterrows():
        print(f"  {row['tag']:40s} {row['count']:>6,} images")
    
    total_instances = filtered_df['count'].sum()
    original_instances = freq_df['count'].sum()
    coverage = (total_instances / original_instances) * 100
    
    print(f"\nCoverage: {coverage:.1f}% of all hair tag instances")
    
    return kept_tags, filtered_df

# ============================================================================
# STEP 2: Filter Images - Keep Only Those With Remaining Tags
# ============================================================================

def filter_images_by_tags(csv_path, kept_tags):
    """
    Keep only images that have at least one of the kept tags.
    
    Args:
        csv_path: Path to original labels.csv
        kept_tags: List of tags to keep
    
    Returns:
        Filtered DataFrame
    """
    print("\n" + "="*70)
    print("STEP 2: Filtering Images")
    print("="*70)
    
    # Load original CSV
    df = pd.read_csv(csv_path)
    print(f"\nOriginal dataset: {len(df)} images")
    
    # Create set for faster lookup
    kept_tags_set = set(kept_tags)
    
    # Filter images
    def has_kept_tag(tags_str):
        """Check if this image has at least one kept tag"""
        if not isinstance(tags_str, str):
            return False
        
        image_tags = tags_str.split()
        # Return True if any tag in this image is in kept_tags
        return any(tag in kept_tags_set for tag in image_tags)
    
    # Apply filter
    filtered_df = df[df['tags'].apply(has_kept_tag)].copy()
    
    print(f"Filtered dataset: {len(filtered_df)} images")
    print(f"Removed: {len(df) - len(filtered_df)} images ({(len(df) - len(filtered_df))/len(df)*100:.1f}%)")
    
    return filtered_df

# ============================================================================
# STEP 3: Clean Tags - Remove Non-Kept Tags from Each Image
# ============================================================================

def clean_image_tags(filtered_df, kept_tags):
    """
    Remove tags that are not in kept_tags from each image's tag string.
    
    Args:
        filtered_df: DataFrame of filtered images
        kept_tags: List of tags to keep
    
    Returns:
        DataFrame with cleaned tags
    """
    print("\n" + "="*70)
    print("STEP 3: Cleaning Tags")
    print("="*70)
    
    kept_tags_set = set(kept_tags)
    
    def clean_tags_str(tags_str):
        """Keep only the tags that are in kept_tags"""
        if not isinstance(tags_str, str):
            return ""
        
        image_tags = tags_str.split()
        # Keep only tags in kept_tags
        cleaned_tags = [tag for tag in image_tags if tag in kept_tags_set]
        return " ".join(cleaned_tags)
    
    # Clean the tags column
    filtered_df['tags'] = filtered_df['tags'].apply(clean_tags_str)
    
    # Count tags per image
    tags_per_image = filtered_df['tags'].apply(lambda x: len(x.split()))
    
    print(f"\nTags per image after cleaning:")
    print(f"  Mean: {tags_per_image.mean():.2f}")
    print(f"  Median: {tags_per_image.median():.0f}")
    print(f"  Min: {tags_per_image.min()}")
    print(f"  Max: {tags_per_image.max()}")
    
    return filtered_df

# ============================================================================
# STEP 4: Create Tag Index (tag_to_idx mapping)
# ============================================================================

def create_tag_index(kept_tags):
    """
    Create tag_to_idx mapping for the model.
    
    Args:
        kept_tags: List of tags to keep
    
    Returns:
        Dictionary mapping tag to index
    """
    print("\n" + "="*70)
    print("STEP 4: Creating Tag Index")
    print("="*70)
    
    # Sort tags alphabetically for consistency
    sorted_tags = sorted(kept_tags)
    
    # Create mapping
    tag_to_idx = {tag: idx for idx, tag in enumerate(sorted_tags)}
    
    print(f"\nCreated tag_to_idx with {len(tag_to_idx)} tags")
    print(f"\nTag mapping (first 10):")
    for i, (tag, idx) in enumerate(list(tag_to_idx.items())[:10]):
        print(f"  {idx:2d}: {tag}")
    
    if len(tag_to_idx) > 10:
        print(f"  ... and {len(tag_to_idx) - 10} more tags")
    
    return tag_to_idx

# ============================================================================
# STEP 5: Save All Outputs
# ============================================================================

def save_outputs(filtered_df, tag_to_idx, kept_tags_df, output_dir):
    """
    Save all output files.
    
    Args:
        filtered_df: Filtered DataFrame with cleaned tags
        tag_to_idx: Tag to index mapping
        kept_tags_df: DataFrame with tag frequencies
        output_dir: Directory to save outputs
    """
    print("\n" + "="*70)
    print("STEP 5: Saving Outputs")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save filtered CSV (filtered images with cleaned tags)
    filtered_csv_path = os.path.join(output_dir, "filtered_labels.csv")
    filtered_df.to_csv(filtered_csv_path, index=False)
    print(f"\n✓ Saved filtered labels to: {filtered_csv_path}")
    print(f"  Contains {len(filtered_df)} images")
    
    # 2. Save tag_to_idx as JSON (for model training)
    tag_index_path = os.path.join(output_dir, "tag_index.json")
    with open(tag_index_path, 'w', encoding='utf-8') as f:
        json.dump(tag_to_idx, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved tag index to: {tag_index_path}")
    print(f"  Contains {len(tag_to_idx)} tags")
    
    # 3. Save tag list (human-readable)
    tag_list_path = os.path.join(output_dir, "tag_list.txt")
    with open(tag_list_path, 'w', encoding='utf-8') as f:
        for tag in sorted(tag_to_idx.keys()):
            f.write(f"{tag}\n")
    print(f"\n✓ Saved tag list to: {tag_list_path}")
    
    # 4. Save tag frequencies
    freq_output_path = os.path.join(output_dir, "kept_tags_frequency.csv")
    kept_tags_df.to_csv(freq_output_path, index=False)
    print(f"\n✓ Saved tag frequencies to: {freq_output_path}")
    
    # 5. Save summary statistics
    stats = {
        "threshold": THRESHOLD,
        "num_tags_kept": len(tag_to_idx),
        "num_images_kept": len(filtered_df),
        "tags": list(tag_to_idx.keys()),
        "tag_frequencies": kept_tags_df.set_index('tag')['count'].to_dict()
    }
    
    stats_path = os.path.join(output_dir, "filter_summary.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved summary to: {stats_path}")
    
    return filtered_csv_path, tag_index_path

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """
    Main filtering workflow
    """
    print("\n" + "="*70)
    print("HAIR TAG FILTERING - FREQUENCY ≥ 400")
    print("="*70)
    
    # Check configuration
    if INPUT_CSV == "path/to/labels.csv":
        print("\n⚠️  ERROR: Please update INPUT_CSV in the script!")
        print("   Edit line 12 with your actual labels.csv path")
        return
    
    print(f"\nConfiguration:")
    print(f"  Input CSV: {INPUT_CSV}")
    print(f"  Frequency CSV: {FREQUENCY_CSV}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    # Step 1: Load frequent tags
    kept_tags, kept_tags_df = load_frequent_tags(FREQUENCY_CSV, THRESHOLD)
    
    # Step 2: Filter images
    filtered_df = filter_images_by_tags(INPUT_CSV, kept_tags)
    
    # Step 3: Clean tags
    filtered_df = clean_image_tags(filtered_df, kept_tags)
    
    # Step 4: Create tag index
    tag_to_idx = create_tag_index(kept_tags)
    
    # Step 5: Save outputs
    filtered_csv, tag_index = save_outputs(filtered_df, tag_to_idx, kept_tags_df, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*70)
    print("FILTERING COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Summary:")
    print(f"  Tags kept: {len(kept_tags)}")
    print(f"  Images kept: {len(filtered_df)}")
    print(f"  Output directory: {OUTPUT_DIR}/")
    
    print(f"\n📁 Output files:")
    print(f"  1. filtered_labels.csv  - Use this CSV for training")
    print(f"  2. tag_index.json       - Tag to index mapping")
    print(f"  3. tag_list.txt         - Human-readable tag list")
    print(f"  4. kept_tags_frequency.csv - Tag statistics")
    print(f"  5. filter_summary.json  - Complete summary")
    
    print(f"\n📋 Next steps:")
    print(f"  1. Review the filtered_labels.csv")
    print(f"  2. Give filtered_labels.csv and tag_index.json to model team")
    print(f"  3. They will use these with their dataset.py")
    
    print(f"\n💡 Usage with their dataset.py:")
    print(f"  dataset = AnimeTagDataset(")
    print(f"      csv_path='{filtered_csv}',")
    print(f"      img_root='images/',")
    print(f"      tag_json_path='{tag_index}',")
    print(f"      split='train'")
    print(f"  )")


if __name__ == "__main__":
    main()
