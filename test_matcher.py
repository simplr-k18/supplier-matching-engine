import pandas as pd
from src.matcher import SupplierMatcher

# Load sample data
df = pd.read_csv('data/raw/sample_suppliers.csv')

print(f"Loaded {len(df)} suppliers")
print(f"\nSample names:")
print(df['vendor_name'].head(10))

# Initialize matcher
matcher = SupplierMatcher(threshold=80)

# Find matches
print("\n🔍 Finding matches...")
matches = matcher.find_matches(df, name_column='vendor_name')

print(f"\n✓ Found {len(matches)} potential matches")
print("\nTop 10 matches:")
print(matches.head(10))

# Save results
matches.to_csv('data/processed/potential_matches.csv', index=False)
print("\n✓ Saved to: data/processed/potential_matches.csv")