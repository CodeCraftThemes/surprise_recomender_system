from knn.knn_train import train_knn_model
from svd.svd_train import train_svd_model
from nmf.nmf_train import train_nmf_model
from slope_one.slope_one_train import train_slope_one
from co_clustering.co_clustering_train import train_co_clustering

datasets = ['ml-100k'] #, 'ml-1m'

for dataset in datasets:
    print(f"\n=== Entrenando modelos para {dataset} ===")
    
    # KNN variants
    train_knn_model('KNNBasic', dataset)
    train_knn_model('KNNWithMeans', dataset)
    
    # Matrix Factorization
    train_svd_model('SVD', dataset)
    train_svd_model('SVDpp', dataset)
    train_nmf_model(dataset)
    
    # Other algorithms
    train_slope_one(dataset)
    train_co_clustering(dataset)

print("\n✅ Todos los modelos entrenados y guardados!")