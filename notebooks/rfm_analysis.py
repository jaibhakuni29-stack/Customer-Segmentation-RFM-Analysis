import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("CUSTOMER SEGMENTATION & RFM ANALYSIS - PROJECT 2")
print("="*80)

# ============================================================================
# SECTION 1: LOAD DATA
# ============================================================================

print("\n[STEP 1] Loading data...")
customers = pd.read_csv('data/customers.csv')
transactions = pd.read_csv('data/transactions.csv')

print(f"✓ Loaded {len(customers)} customers")
print(f"✓ Loaded {len(transactions)} transactions")

# ============================================================================
# SECTION 2: MERGE DATA
# ============================================================================

print("\n[STEP 2] Merging customer and transaction data...")
df = transactions.merge(customers, on='Customer_ID', how='left')
print(f"✓ Merged data shape: {df.shape}")

# Convert dates
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
customers['Signup_Date'] = pd.to_datetime(customers['Signup_Date'])

# ============================================================================
# SECTION 3: RFM ANALYSIS
# ============================================================================

print("\n[STEP 3] Calculating RFM metrics...")

# Reference date (latest date in data)
reference_date = df['Transaction_Date'].max() + pd.Timedelta(days=1)
print(f"  Reference date: {reference_date.date()}")

# Calculate RFM for each customer
rfm = df.groupby('Customer_ID').agg({
    'Transaction_Date': lambda x: (reference_date - x.max()).days,  # Recency
    'Transaction_ID': 'count',  # Frequency
    'Amount': 'sum'  # Monetary
}).rename(columns={
    'Transaction_Date': 'Recency',
    'Transaction_ID': 'Frequency',
    'Amount': 'Monetary'
})

print(f"✓ RFM calculated for {len(rfm)} customers")

# Display RFM statistics
print("\n[STEP 4] RFM Statistics:")
print(rfm.describe())

# ============================================================================
# SECTION 5: RFM SCORING (1-5 scale)
# ============================================================================

print("\n[STEP 5] Creating RFM scores (1-5 scale)...")

# Recency: Lower is better (recently purchased = higher score)
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')

# Frequency: Higher is better (more purchases = higher score)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# Monetary: Higher is better (more spending = higher score)
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# Convert to numeric
rfm['R_Score'] = pd.to_numeric(rfm['R_Score'])
rfm['F_Score'] = pd.to_numeric(rfm['F_Score'])
rfm['M_Score'] = pd.to_numeric(rfm['M_Score'])

# Combined RFM Score
rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

print("✓ RFM scores created")
print(f"\nRFM Score Distribution:")
print(rfm['RFM_Score'].value_counts().sort_index(ascending=False))

# ============================================================================
# SECTION 6: K-MEANS CLUSTERING
# ============================================================================

print("\n[STEP 6] Preparing data for clustering...")

# Select features for clustering
X = rfm[['Recency', 'Frequency', 'Monetary']].values

# Standardize features (important for K-Means!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✓ Data standardized")

# ============================================================================
# SECTION 7: FIND OPTIMAL NUMBER OF CLUSTERS (Elbow Method)
# ============================================================================

print("\n[STEP 7] Finding optimal number of clusters...")

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    print(f"  K={k}: Inertia={kmeans.inertia_:.2f}")

print("✓ Clustering tested for K=2 to K=10")

# ============================================================================
# SECTION 8: VISUALIZE ELBOW CURVE
# ============================================================================

print("\n[STEP 8] Creating Elbow curve visualization...")

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.title('Elbow Method - Optimal Number of Clusters', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/01_elbow_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_elbow_curve.png")
plt.close()

# ============================================================================
# SECTION 9: TRAIN FINAL MODEL (K=4)
# ============================================================================

print("\n[STEP 9] Training K-Means with K=4...")

optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans_final.fit_predict(X_scaled)

print(f"✓ Model trained with {optimal_k} clusters")
print(f"\nCluster Distribution:")
print(rfm['Cluster'].value_counts().sort_index())

# ============================================================================
# SECTION 10: ANALYZE CLUSTERS
# ============================================================================

print("\n[STEP 10] Analyzing cluster characteristics...")

cluster_analysis = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'RFM_Score': 'mean'
}).round(2)

print("\nCluster Profiles:")
print(cluster_analysis)

# ============================================================================
# SECTION 11: ASSIGN SEGMENT NAMES
# ============================================================================

print("\n[STEP 11] Assigning segment names...")

def assign_segment(row):
    """Assign meaningful segment names based on RFM scores"""
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    
    # Champions: High R, F, M
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    # Loyal: High F and M, Good R
    elif f >= 4 and m >= 4:
        return 'Loyal Customers'
    # Potential Loyalists: Good scores, Recent
    elif r >= 4 and f >= 3 and m >= 3:
        return 'Potential Loyalists'
    # At Risk: Low R, but good F and M
    elif r <= 2 and f >= 3 and m >= 3:
        return 'At Risk'
    # Lost: Low R and F
    elif r <= 2 and f <= 2:
        return 'Lost Customers'
    else:
        return 'Others'

rfm['Segment'] = rfm.apply(assign_segment, axis=1)

print(f"✓ Segments assigned")
print(f"\nSegment Distribution:")
print(rfm['Segment'].value_counts())

# ============================================================================
# SECTION 12: CALCULATE CUSTOMER LIFETIME VALUE (CLV)
# ============================================================================

print("\n[STEP 12] Calculating Customer Lifetime Value...")

rfm['CLV'] = rfm['Frequency'] * rfm['Monetary']

print(f"✓ CLV calculated")
print(f"\nCLV Statistics:")
print(rfm['CLV'].describe())

# ============================================================================
# SECTION 13: VISUALIZATION 1 - CLUSTER SCATTER PLOT
# ============================================================================

print("\n[STEP 13] Creating cluster visualization...")

fig = plt.figure(figsize=(14, 10))

# 3D scatter plot of clusters
ax = fig.add_subplot(2, 2, 1, projection='3d')
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
for cluster in range(optimal_k):
    cluster_data = rfm[rfm['Cluster'] == cluster]
    ax.scatter(cluster_data['Recency'], 
               cluster_data['Frequency'], 
               cluster_data['Monetary'],
               c=colors[cluster], 
               label=f'Cluster {cluster}',
               s=100, 
               alpha=0.6)

ax.set_xlabel('Recency (days)', fontsize=10)
ax.set_ylabel('Frequency (purchases)', fontsize=10)
ax.set_zlabel('Monetary (₹)', fontsize=10)
ax.set_title('3D Cluster Visualization', fontsize=12, fontweight='bold')
ax.legend()

# 2D: Recency vs Frequency
ax2 = fig.add_subplot(2, 2, 2)
for cluster in range(optimal_k):
    cluster_data = rfm[rfm['Cluster'] == cluster]
    ax2.scatter(cluster_data['Recency'], 
                cluster_data['Frequency'],
                c=colors[cluster], 
                label=f'Cluster {cluster}',
                s=100, 
                alpha=0.6)
ax2.set_xlabel('Recency (days)', fontsize=10)
ax2.set_ylabel('Frequency (purchases)', fontsize=10)
ax2.set_title('Recency vs Frequency', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 2D: Frequency vs Monetary
ax3 = fig.add_subplot(2, 2, 3)
for cluster in range(optimal_k):
    cluster_data = rfm[rfm['Cluster'] == cluster]
    ax3.scatter(cluster_data['Frequency'], 
                cluster_data['Monetary'],
                c=colors[cluster], 
                label=f'Cluster {cluster}',
                s=100, 
                alpha=0.6)
ax3.set_xlabel('Frequency (purchases)', fontsize=10)
ax3.set_ylabel('Monetary (₹)', fontsize=10)
ax3.set_title('Frequency vs Monetary', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 2D: Recency vs Monetary
ax4 = fig.add_subplot(2, 2, 4)
for cluster in range(optimal_k):
    cluster_data = rfm[rfm['Cluster'] == cluster]
    ax4.scatter(cluster_data['Recency'], 
                cluster_data['Monetary'],
                c=colors[cluster], 
                label=f'Cluster {cluster}',
                s=100, 
                alpha=0.6)
ax4.set_xlabel('Recency (days)', fontsize=10)
ax4.set_ylabel('Monetary (₹)', fontsize=10)
ax4.set_title('Recency vs Monetary', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/02_cluster_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_cluster_visualization.png")
plt.close()

# ============================================================================
# SECTION 14: VISUALIZATION 2 - SEGMENT DISTRIBUTION
# ============================================================================

print("\n[STEP 14] Creating segment distribution charts...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Segment counts
segment_counts = rfm['Segment'].value_counts()
axes[0].bar(segment_counts.index, segment_counts.values, color='skyblue', edgecolor='black')
axes[0].set_title('Customer Distribution by Segment', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Customers')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# Segment revenue
segment_revenue = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
axes[1].barh(segment_revenue.index, segment_revenue.values, color='lightcoral', edgecolor='black')
axes[1].set_title('Revenue Contribution by Segment', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Total Revenue (₹)')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('visualizations/03_segment_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_segment_distribution.png")
plt.close()

# ============================================================================
# SECTION 15: VISUALIZATION 3 - RFM HEATMAP
# ============================================================================

print("\n[STEP 15] Creating RFM characteristics heatmap...")

segment_rfm = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(segment_rfm.T, annot=True, fmt='.1f', cmap='RdYlGn_r', 
            cbar_kws={'label': 'Average Value'}, linewidths=1)
plt.title('RFM Characteristics by Segment', fontsize=14, fontweight='bold')
plt.xlabel('Segment', fontsize=12)
plt.ylabel('RFM Metric', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/04_rfm_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_rfm_heatmap.png")
plt.close()

# ============================================================================
# SECTION 16: VISUALIZATION 4 - CLV DISTRIBUTION
# ============================================================================

print("\n[STEP 16] Creating Customer Lifetime Value visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CLV by segment
clv_by_segment = rfm.groupby('Segment')['CLV'].mean().sort_values(ascending=False)
axes[0].bar(clv_by_segment.index, clv_by_segment.values, color='gold', edgecolor='black')
axes[0].set_title('Average CLV by Segment', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Average CLV (₹)')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# CLV distribution
axes[1].hist(rfm['CLV'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].set_title('Distribution of Customer Lifetime Value', fontsize=12, fontweight='bold')
axes[1].set_xlabel('CLV (₹)')
axes[1].set_ylabel('Number of Customers')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visualizations/05_clv_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_clv_analysis.png")
plt.close()

# ============================================================================
# SECTION 17: BUSINESS RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("BUSINESS RECOMMENDATIONS BY SEGMENT")
print("="*80)

recommendations = {
    'Champions': {
        'count': len(rfm[rfm['Segment'] == 'Champions']),
        'revenue': rfm[rfm['Segment'] == 'Champions']['Monetary'].sum(),
        'strategy': [
            '✓ VIP treatment: Exclusive offers and early access to new products',
            '✓ Loyalty rewards: Special discounts, points, and perks',
            '✓ Personal account manager for premium support',
            '✓ Invite to VIP events and product launches',
            '✓ Referral incentives: Reward them for bringing new customers'
        ]
    },
    'Loyal Customers': {
        'count': len(rfm[rfm['Segment'] == 'Loyal Customers']),
        'revenue': rfm[rfm['Segment'] == 'Loyal Customers']['Monetary'].sum(),
        'strategy': [
            '✓ Maintain engagement: Regular personalized offers',
            '✓ Cross-sell and upsell: Introduce complementary products',
            '✓ Birthday and anniversary discounts',
            '✓ Exclusive member-only sales',
            '✓ Loyalty program with tiered benefits'
        ]
    },
    'Potential Loyalists': {
        'count': len(rfm[rfm['Segment'] == 'Potential Loyalists']),
        'revenue': rfm[rfm['Segment'] == 'Potential Loyalists']['Monetary'].sum(),
        'strategy': [
            '✓ Nurture relationship: Regular engagement campaigns',
            '✓ Increase purchase frequency: Limited-time offers',
            '✓ Product recommendations based on purchase history',
            '✓ Welcome series with educational content',
            '✓ Feedback surveys to understand preferences'
        ]
    },
    'At Risk': {
        'count': len(rfm[rfm['Segment'] == 'At Risk']),
        'revenue': rfm[rfm['Segment'] == 'At Risk']['Monetary'].sum(),
        'strategy': [
            '✓ Win-back campaigns: Special discounts to re-engage',
            '✓ Personalized outreach: "We miss you" emails',
            '✓ Offer exclusive deals they can\'t refuse',
            '✓ Survey to understand why they\'re not purchasing',
            '✓ Incentivize next purchase with bonus points'
        ]
    },
    'Lost Customers': {
        'count': len(rfm[rfm['Segment'] == 'Lost Customers']),
        'revenue': rfm[rfm['Segment'] == 'Lost Customers']['Monetary'].sum(),
        'strategy': [
            '✓ Re-activation campaigns: Deep discounts',
            '✓ Email outreach highlighting what\'s new',
            '✓ One-time offer with expiration date',
            '✓ Request feedback: Why did they leave?',
            '✓ Consider removing from list if no response'
        ]
    }
}

for segment, info in recommendations.items():
    if info['count'] > 0:
        print(f"\n{segment.upper()}")
        print(f"  Customers: {info['count']}")
        print(f"  Total Revenue: ₹{info['revenue']:,.2f}")
        print(f"  Recommended Strategy:")
        for strategy in info['strategy']:
            print(f"    {strategy}")

# ============================================================================
# SECTION 18: SAVE RFM RESULTS
# ============================================================================

print("\n[STEP 17] Saving RFM analysis results...")

# Add customer info
rfm_with_customer_info = rfm.copy()
rfm_with_customer_info = rfm_with_customer_info.reset_index()

# Save to CSV
rfm_with_customer_info.to_csv('data/rfm_analysis_results.csv', index=False)
print("✓ Saved: rfm_analysis_results.csv")

# ============================================================================
# SECTION 19: SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ RFM ANALYSIS & CUSTOMER SEGMENTATION COMPLETE!")
print("="*80)

print(f"\nKey Metrics:")
print(f"  • Total Customers: {len(rfm)}")
print(f"  • Total Revenue: ₹{rfm['Monetary'].sum():,.2f}")
print(f"  • Average Order Value: ₹{rfm['Monetary'].mean():,.2f}")
print(f"  • Average CLV: ₹{rfm['CLV'].mean():,.2f}")
print(f"\nSegment Summary:")
for segment in rfm['Segment'].unique():
    segment_data = rfm[rfm['Segment'] == segment]
    print(f"  • {segment}: {len(segment_data)} customers, Revenue: ₹{segment_data['Monetary'].sum():,.2f}")

print(f"\nVisualizations saved:")
print(f"  ✓ 01_elbow_curve.png")
print(f"  ✓ 02_cluster_visualization.png")
print(f"  ✓ 03_segment_distribution.png")
print(f"  ✓ 04_rfm_heatmap.png")
print(f"  ✓ 05_clv_analysis.png")

print(f"\nData saved:")
print(f"  ✓ rfm_analysis_results.csv")

print("\n" + "="*80)