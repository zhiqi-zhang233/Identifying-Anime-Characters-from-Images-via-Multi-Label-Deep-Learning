"""
Hair Tag Analysis Script
For data preprocessing - analyzes and combines similar hair tags

Author: Data Team
Purpose: Explore hair tags, find similar ones, and determine frequency thresholds
"""

import pandas as pd
from collections import Counter, defaultdict
import json
import re

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
CSV_PATH = "labels_combined.csv"  # UPDATE THIS!
OUTPUT_DIR = "/Users/apple/Desktop/Fall_2025/DS5660-DL/mini_project/data_fetch"

# ============================================================================
# STEP 1: Extract and analyze all hair-related tags
# ============================================================================

def extract_hair_tags(csv_path):
    """
    Extract all hair-related tags from the CSV
    Returns a Counter object with tag frequencies
    """
    print("="*70)
    print("STEP 1: Extracting Hair Tags")
    print("="*70)
    
    # Load CSV
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} images")
    
    # Keywords that indicate hair-related tags
    hair_keywords = [
        'hair', 'ponytail', 'twintails', 'braid', 'bun', 'bangs', 
        'ahoge', 'sidelocks', 'hairband', 'hair_ribbon', 'hair_bow', 
        'hair_ornament', 'hair_flower', 'hairclip', 'scrunchie', 
        'hair_tie', 'hair_bobbles', 'hairpin'
    ]
    
    # Extract all tags
    print("\nExtracting tags from all images...")
    all_hair_tags = []
    
    for idx, row in df.iterrows():
        tags = str(row['tags']).split()
        
        # Filter for hair-related tags
        for tag in tags:
            if any(keyword in tag.lower() for keyword in hair_keywords):
                all_hair_tags.append(tag)
    
    # Count frequencies
    tag_frequencies = Counter(all_hair_tags)
    
    print(f"\n✓ Found {len(all_hair_tags)} hair tag instances")
    print(f"✓ Found {len(tag_frequencies)} unique hair tags")
    
    return tag_frequencies, df

# ============================================================================
# STEP 2: Group similar tags together
# ============================================================================

def group_similar_tags(tag_frequencies):
    """
    Group similar tags together based on patterns
    For example: pink_hair, light_pink_hair -> pink_hair family
    """
    print("\n" + "="*70)
    print("STEP 2: Grouping Similar Tags")
    print("="*70)
    
    # Define tag categories and their variations
    tag_groups = {
        # Hair Colors
        'pink_hair': ['pink_hair', 'light_pink_hair', 'dark_pink_hair', 'pastel_pink_hair'],
        'blue_hair': ['blue_hair', 'light_blue_hair', 'dark_blue_hair', 'aqua_hair', 'sky_blue_hair'],
        'blonde_hair': ['blonde_hair', 'light_blonde_hair', 'platinum_blonde_hair', 'golden_hair'],
        'brown_hair': ['brown_hair', 'light_brown_hair', 'dark_brown_hair'],
        'black_hair': ['black_hair', 'dark_hair'],
        'white_hair': ['white_hair', 'silver_hair', 'grey_hair', 'gray_hair'],
        'red_hair': ['red_hair', 'dark_red_hair', 'orange_hair', 'ginger_hair'],
        'green_hair': ['green_hair', 'light_green_hair', 'dark_green_hair'],
        'purple_hair': ['purple_hair', 'light_purple_hair', 'dark_purple_hair', 'violet_hair'],
        'multicolored_hair': ['multicolored_hair', 'two-tone_hair', 'gradient_hair', 'colored_inner_hair'],
        
        # Hair Length
        'long_hair': ['long_hair', 'very_long_hair', 'absurdly_long_hair'],
        'short_hair': ['short_hair', 'very_short_hair'],
        'medium_hair': ['medium_hair', 'shoulder_length_hair'],
        
        # Hair Styles
        'ponytail': ['ponytail', 'high_ponytail', 'low_ponytail', 'side_ponytail'],
        'twintails': ['twintails', 'low_twintails'],
        'braid': ['braid', 'single_braid', 'side_braid', 'french_braid', 'twin_braids'],
        'bun': ['single_hair_bun', 'double_bun', 'hair_bun', 'cone_hair_bun'],
        
        # Hair Accessories
        'hair_ribbon': ['hair_ribbon', 'hair_bow'],
        'hairband': ['hairband', 'headband'],
        'hair_ornament': ['hair_ornament', 'hair_flower', 'hair_bell'],
        'hairclip': ['hairclip', 'hairpin', 'hair_bobbles'],
        
        # Hair Features
        'bangs': ['bangs', 'blunt_bangs', 'swept_bangs', 'parted_bangs'],
        'ahoge': ['ahoge', 'antenna_hair'],
        'sidelocks': ['sidelocks', 'long_sidelocks'],
    }
    
    # Find which tags in our data match these groups
    grouped_counts = defaultdict(list)
    ungrouped_tags = []
    
    for tag, count in tag_frequencies.items():
        matched = False
        for group_name, variations in tag_groups.items():
            if tag in variations:
                grouped_counts[group_name].append((tag, count))
                matched = True
                break
        
        if not matched:
            ungrouped_tags.append((tag, count))
    
    # Display grouped results
    print("\n📊 GROUPED HAIR TAGS (by category):\n")
    
    total_grouped = 0
    for group_name, tags_in_group in sorted(grouped_counts.items()):
        total_count = sum(count for _, count in tags_in_group)
        total_grouped += total_count
        print(f"\n🔹 {group_name.upper().replace('_', ' ')} (Total: {total_count})")
        for tag, count in sorted(tags_in_group, key=lambda x: x[1], reverse=True):
            print(f"   • {tag:40s} {count:>6,} images")
    
    print(f"\n\n📊 UNGROUPED HAIR TAGS ({len(ungrouped_tags)} tags):\n")
    for tag, count in sorted(ungrouped_tags, key=lambda x: x[1], reverse=True)[:30]:
        print(f"   • {tag:40s} {count:>6,} images")
    
    if len(ungrouped_tags) > 30:
        print(f"\n   ... and {len(ungrouped_tags) - 30} more ungrouped tags")
    
    return grouped_counts, ungrouped_tags

# ============================================================================
# STEP 3: Create frequency report
# ============================================================================

def create_frequency_report(tag_frequencies, grouped_counts, ungrouped_tags):
    """
    Create detailed frequency report for review
    """
    print("\n" + "="*70)
    print("STEP 3: Creating Frequency Report")
    print("="*70)
    
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. All individual tags sorted by frequency
    all_tags_report = []
    for tag, count in tag_frequencies.most_common():
        all_tags_report.append({
            'tag': tag,
            'count': count
        })
    
    # Save as CSV
    df_all = pd.DataFrame(all_tags_report)
    csv_path = f"{OUTPUT_DIR}/all_hair_tags_frequency.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\n✓ Saved all tags to: {csv_path}")
    
    # 2. Grouped tags report
    grouped_report = []
    for group_name, tags_in_group in sorted(grouped_counts.items()):
        total_count = sum(count for _, count in tags_in_group)
        grouped_report.append({
            'group': group_name,
            'total_count': total_count,
            'num_variations': len(tags_in_group),
            'variations': ', '.join([f"{tag}({count})" for tag, count in tags_in_group])
        })
    
    df_grouped = pd.DataFrame(grouped_report)
    df_grouped = df_grouped.sort_values('total_count', ascending=False)
    csv_path = f"{OUTPUT_DIR}/grouped_hair_tags_frequency.csv"
    df_grouped.to_csv(csv_path, index=False)
    print(f"✓ Saved grouped tags to: {csv_path}")
    
    # 3. Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   Total unique hair tags: {len(tag_frequencies)}")
    print(f"   Total hair tag instances: {sum(tag_frequencies.values()):,}")
    print(f"   Tags grouped into categories: {len(grouped_counts)}")
    print(f"   Ungrouped tags: {len(ungrouped_tags)}")
    
    # 4. Frequency distribution
    frequencies = list(tag_frequencies.values())
    print(f"\n📊 FREQUENCY DISTRIBUTION:")
    thresholds = [1, 10, 50, 100, 200, 500, 1000]
    for i, thresh in enumerate(thresholds):
        if i < len(thresholds) - 1:
            count = sum(1 for f in frequencies if thresh <= f < thresholds[i+1])
            print(f"   {thresh:>4} - {thresholds[i+1]:>4} occurrences: {count:>4} tags")
        else:
            count = sum(1 for f in frequencies if f >= thresh)
            print(f"   {thresh:>4}+       occurrences: {count:>4} tags")
    
    # 5. Recommendations
    print(f"\n💡 RECOMMENDATIONS FOR THRESHOLD:")
    for threshold in [50, 100, 150, 200, 300]:
        tags_above = sum(1 for f in frequencies if f >= threshold)
        coverage = sum(f for f in frequencies if f >= threshold) / sum(frequencies) * 100
        print(f"   Threshold = {threshold:>3}: Keep {tags_above:>3} tags (covers {coverage:.1f}% of data)")

# ============================================================================
# STEP 4: Interactive threshold selection
# ============================================================================

def save_filtered_tags(tag_frequencies, threshold):
    """
    Save filtered tags based on threshold
    """
    print(f"\n" + "="*70)
    print(f"FILTERING TAGS WITH THRESHOLD = {threshold}")
    print("="*70)
    
    # Filter tags
    filtered_tags = {tag: count for tag, count in tag_frequencies.items() 
                    if count >= threshold}
    
    print(f"\n✓ Kept {len(filtered_tags)} tags (out of {len(tag_frequencies)} total)")
    print(f"✓ Coverage: {sum(filtered_tags.values())/sum(tag_frequencies.values())*100:.1f}% of all hair tag instances")
    
    # Show top tags
    print(f"\n📋 TOP 30 TAGS AFTER FILTERING:\n")
    for i, (tag, count) in enumerate(sorted(filtered_tags.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True)[:30], 1):
        print(f"   {i:2d}. {tag:40s} {count:>6,} images")
    
    # Save to JSON
    output_file = f"{OUTPUT_DIR}/filtered_tags_threshold_{threshold}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'threshold': threshold,
            'num_tags': len(filtered_tags),
            'tags': filtered_tags,
            'tag_list': list(filtered_tags.keys())
        }, f, indent=2)
    
    print(f"\n✓ Saved filtered tags to: {output_file}")
    
    return filtered_tags

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution flow
    """
    print("\n" + "="*70)
    print("HAIR TAG ANALYSIS FOR DATA PREPROCESSING")
    print("="*70)
    
    # Check if path is updated
    if CSV_PATH == "/path/to/your/tags.csv":
        print("\n⚠️  ERROR: Please update CSV_PATH in this script!")
        print("   Edit line 14 with your actual CSV file path")
        return
    
    # Step 1: Extract hair tags
    tag_frequencies, df = extract_hair_tags(CSV_PATH)
    
    # Step 2: Group similar tags
    grouped_counts, ungrouped_tags = group_similar_tags(tag_frequencies)
    
    # Step 3: Create frequency report
    create_frequency_report(tag_frequencies, grouped_counts, ungrouped_tags)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"\nFiles created:")
    print(f"   1. all_hair_tags_frequency.csv     - All tags with counts")
    print(f"   2. grouped_hair_tags_frequency.csv - Grouped tags by category")
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Review the CSV files in {OUTPUT_DIR}")
    print(f"   2. Look at the recommendations above")
    print(f"   3. Decide on a threshold (e.g., 100)")
    print(f"   4. Run: python tag_analysis.py --threshold 100")
    print(f"      (or edit this script and call save_filtered_tags manually)")

if __name__ == "__main__":
    import sys
    
    # Check if threshold provided as argument
    if len(sys.argv) > 2 and sys.argv[1] == '--threshold':
        threshold = int(sys.argv[2])
        print(f"\nRunning with threshold = {threshold}")
        tag_frequencies, df = extract_hair_tags(CSV_PATH)
        filtered_tags = save_filtered_tags(tag_frequencies, threshold)
    else:
        # Run full analysis
        main()