import hdbscan
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
import umap
import seaborn as sns


def clustering_goat(df, col):
    statements = df.iloc[:, col].tolist()
    # --- Step 1: Create embeddings ---
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(statements, show_progress_bar=True)
    # --- Step 2: Apply HDBSCAN ---
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=12,
        min_samples=5,
        metric='euclidean'
    )
    clusters = clusterer.fit_predict(embeddings)
    df['cluster'] = clusters
    # --- Step 3: Quick cluster summary ---
    print("Cluster counts (including noise -1):")
    print(df["cluster"].value_counts())

    # --- Step 4: Optional 2D visualization using UMAP ---
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(12,8))
    sns.scatterplot(
        x=embedding_2d[:,0],
        y=embedding_2d[:,1],
        hue=df["cluster"],
        palette="tab20",
        legend="full",
        s=10
    )
    plt.title("HDBSCAN clusters of statements (2D UMAP projection)")
    plt.show()
    print(df.groupby("cluster")["label"].value_counts())
    for cluster_id in df["cluster"].unique():
        print(f"\nCluster {cluster_id} examples:")
        print(df[df["cluster"] == cluster_id]["statement"].head(5))


    return