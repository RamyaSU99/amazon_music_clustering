# ==========================================
# 🎵 Amazon Music Clustering – Streamlit App
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px

# ------------------------------------------------------------
# 1️⃣ PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Amazon Music Cluster Visualizer", layout="wide")

st.title("🎧 Amazon Music Clustering Dashboard")
st.markdown("""
Explore how songs are grouped based on their audio and artist features using  
*K-Means, **DBSCAN, and **Hierarchical Clustering*.
---
""")

# ------------------------------------------------------------
# 2️⃣ LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/thangayogeshk/Desktop/amazon music clustering/amazon_music_clusters_all_methods.csv")
    return df

df = load_data()

st.sidebar.header("⚙ Filter Options")
method = st.sidebar.selectbox("Select Clustering Method", ["K-Means", "DBSCAN", "Hierarchical"])
cluster_col = {
    "K-Means": "cluster",
    "DBSCAN": "cluster_dbscan",
    "Hierarchical": "cluster_hc"
}[method]

# Drop NaN clusters (for HC/DBSCAN)
df_vis = df.dropna(subset=[cluster_col]).copy()
df_vis[cluster_col] = df_vis[cluster_col].astype(int)

# ------------------------------------------------------------
# 3️⃣ DATA OVERVIEW
# ------------------------------------------------------------
st.subheader("📊 Dataset Overview")

# Show preview of top rows
with st.expander("🔍 Preview Dataset (Top 10 Rows)", expanded=False):
    st.dataframe(df_vis.head(10).style.format(precision=2))

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("🎵 Number of Songs", f"{len(df_vis):,}")
col2.metric("🧠 Number of Clusters", df_vis[cluster_col].nunique())
col3.metric("🎼 Unique Genres", df_vis['genres'].nunique())

# Cluster feature summary
st.markdown("### 📋 Cluster Summary (Mean Feature Values)")
features = ['danceability', 'energy', 'valence', 'tempo']
cluster_summary = df_vis.groupby(cluster_col)[features].mean()
st.dataframe(cluster_summary.style.background_gradient(cmap='YlGnBu').format("{:.2f}"))

# Optional pie chart
with st.expander("📊 Cluster Size – Pie Chart", expanded=False):
    cluster_counts = df_vis[cluster_col].value_counts().sort_index()
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(
        cluster_counts,
        labels=cluster_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("pastel")
    )
    ax_pie.axis("equal")
    ax_pie.set_title(f"{method} – Cluster Size Proportion")
    st.pyplot(fig_pie)

# ------------------------------------------------------------
# 4️⃣ CLUSTER DISTRIBUTION
# ------------------------------------------------------------
st.subheader(f"🎨 {method} Cluster Distribution")
cluster_counts = df_vis[cluster_col].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax, palette="viridis")
ax.set_xlabel("Cluster ID")
ax.set_ylabel("Number of Songs")
ax.set_title(f"{method} – Cluster Size Distribution")
st.pyplot(fig)

# ------------------------------------------------------------
# 5️⃣ FEATURE COMPARISON PER CLUSTER
# ------------------------------------------------------------
st.subheader("🎼 Average Feature Values per Cluster")

cluster_profile = df_vis.groupby(cluster_col)[features].mean()

fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(cluster_profile, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax)
ax.set_title(f"{method} – Feature Profile Heatmap")
st.pyplot(fig)

# ------------------------------------------------------------
# 6️⃣ PCA VISUALIZATION (2D)
# ------------------------------------------------------------
st.subheader("🌀 PCA Visualization (2D Projection)")

# Run PCA on numeric features
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_vis[features])
df_vis['pca1'], df_vis['pca2'] = pca_result[:,0], pca_result[:,1]

explained_var = pca.explained_variance_ratio_.sum()
st.markdown(f"🧮 PCA captures **{explained_var:.2%}** of variance in selected features.")

fig_pca = px.scatter(
    df_vis, x='pca1', y='pca2',
    color=df_vis[cluster_col].astype(str),
    hover_data=['genres'],
    title=f"{method} – PCA Cluster Visualization",
    color_discrete_sequence=px.colors.qualitative.Vivid
)
st.plotly_chart(fig_pca, use_container_width=True)

# ------------------------------------------------------------
# 7️⃣ GENRE CLUSTER DISTRIBUTION
# ------------------------------------------------------------
st.subheader("🎶 Top Genres by Cluster")

# Clean genres and drop missing ones
df_vis['genres'] = df_vis['genres'].astype(str).str.title().str.strip()
df_vis = df_vis[df_vis['genres'].notna()]

top_genres = (
    df_vis.groupby(['genres', cluster_col])
    .size()
    .reset_index(name='count')
    .sort_values('count', ascending=False)
    .head(15)
)

fig_genres = px.bar(
    top_genres, 
    x='genres', y='count', color=cluster_col,
    title=f"Top Genres across {method} Clusters",
    color_discrete_sequence=px.colors.qualitative.Bold
)
st.plotly_chart(fig_genres, use_container_width=True)

# ------------------------------------------------------------
# 8️⃣ DOWNLOAD FILTERED RESULTS
# ------------------------------------------------------------
st.subheader("⬇ Download Filtered Cluster Data")
csv = df_vis.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download CSV",
    csv,
    f"music_clusters_{method.lower()}.csv",
    "text/csv",
    key='download-csv'
)

st.markdown("---")
st.caption("Built with ❤ using Streamlit | Dataset: Amazon Music Single-Genre Artists")