import pandas as pd
import random
from datetime import datetime, timedelta

# Sample company names with variations (like real procurement data)
companies_base = [
    "Apple Inc.",
    "Microsoft Corporation", 
    "Amazon.com Inc.",
    "Google LLC",
    "Meta Platforms Inc.",
    "Tesla Inc.",
    "JPMorgan Chase & Co.",
    "Johnson & Johnson",
    "Procter & Gamble Co.",
    "Coca-Cola Company",
    "PepsiCo Inc.",
    "Intel Corporation",
    "IBM Corporation",
    "Oracle Corporation",
    "Salesforce Inc."
]

# Generate variations (this simulates messy real data)
def generate_variations(company_name):
    variations = [company_name]  # Original
    
    # Common variations
    variations.append(company_name.replace("Inc.", "").strip())
    variations.append(company_name.replace("Corporation", "Corp").strip())
    variations.append(company_name.replace("Company", "Co").strip())
    variations.append(company_name.replace("&", "and"))
    variations.append(company_name.upper())
    variations.append(company_name.lower())
    variations.append(company_name.replace(".", ""))
    variations.append(company_name.replace(",", ""))
    
    # Typos
    if "Apple" in company_name:
        variations.extend(["Aple Inc", "Apple Computers", "APPLE INC"])
    if "Microsoft" in company_name:
        variations.extend(["Microsft Corp", "MS Corporation", "MSFT"])
    if "Amazon" in company_name:
        variations.extend(["Amazon.com", "Amazon", "AMZN"])
    if "Google" in company_name:
        variations.extend(["Google Inc", "Alphabet Inc", "GOOGL"])
        
    return variations

# Generate dataset
records = []
record_id = 1

for company in companies_base:
    variations = generate_variations(company)
    
    # Create 3-5 records per company with different variations
    for _ in range(random.randint(3, 5)):
        variation = random.choice(variations)
        
        records.append({
            'vendor_id': f'V{record_id:05d}',
            'vendor_name': variation,
            'country': random.choice(['USA', 'UK', 'Canada', 'Germany', 'France']),
            'spend_amount': random.randint(10000, 1000000),
            'invoice_count': random.randint(1, 50),
            'first_transaction_date': (datetime.now() - timedelta(days=random.randint(30, 730))).strftime('%Y-%m-%d'),
            'category': random.choice(['Technology', 'Services', 'Manufacturing', 'Consulting'])
        })
        record_id += 1

# Create DataFrame
df = pd.DataFrame(records)

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Save
df.to_csv('data/raw/sample_suppliers.csv', index=False)
print(f"✓ Generated {len(df)} supplier records")
print(f"✓ {len(companies_base)} unique parent companies")
print(f"✓ Saved to: data/raw/sample_suppliers.csv")

# Show sample
print("\nSample data:")
print(df.head(10))