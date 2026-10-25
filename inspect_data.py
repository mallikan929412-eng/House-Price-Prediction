import csv

with open('data/bengaluru_house_prices_15000.csv', mode='r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

for r in rows:
    p = float(r['price_lakhs']) * 100000
    a = float(r['area_sqft'])
    r['pps'] = p / a

# Group by locality to check locality-specific IQR or overall IQR
localities = sorted(list(set(r['locality'] for r in rows)))

pps_all = sorted([r['pps'] for r in rows])
n = len(pps_all)
q1 = pps_all[int(n*0.25)]
q3 = pps_all[int(n*0.75)]
iqr = q3 - q1
upper_1_5 = q3 + 1.5 * iqr
lower_1_5 = q1 - 1.5 * iqr

# Let's check locality-based IQR
outliers_locality = []
for loc in localities:
    loc_rows = [r for r in rows if r['locality'] == loc]
    loc_pps = sorted([r['pps'] for r in loc_rows])
    l_n = len(loc_pps)
    l_q1 = loc_pps[int(l_n*0.25)]
    l_q3 = loc_pps[int(l_n*0.75)]
    l_iqr = l_q3 - l_q1
    l_upper = l_q3 + 1.5 * l_iqr
    l_lower = max(0, l_q1 - 1.5 * l_iqr)
    loc_outliers = [r for r in loc_rows if r['pps'] > l_upper or r['pps'] < l_lower]
    outliers_locality.extend(loc_outliers)

print(f"Total rows: {len(rows)}")
print(f"Global 1.5 IQR upper: {upper_1_5:.1f}, lower: {lower_1_5:.1f}")
global_outliers = [r for r in rows if r['pps'] > upper_1_5 or r['pps'] < lower_1_5]
print(f"Global IQR outliers count: {len(global_outliers)}")
print(f"Locality-level IQR outliers count: {len(outliers_locality)}")

# Let's check 2.5 or 3.0 IQR or where the ~40 odd prices are
for k in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]:
    cnt = sum(1 for r in rows if r['pps'] > q3 + k*iqr or r['pps'] < q1 - k*iqr)
    print(f"k={k}: {cnt} outliers")
