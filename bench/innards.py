import numpy as np

for name in ["rootwise", "zonewise"]:
    xb = np.load(f"/Users/vihanpatil/school/captsone/RootWise-2.0/evaluation/faiss_bench_data/{name}/xb.npy")
    xq = np.load(f"/Users/vihanpatil/school/captsone/RootWise-2.0/evaluation/faiss_bench_data/{name}/xq.npy")
    xtrain = np.load(f"/Users/vihanpatil/school/captsone/RootWise-2.0/evaluation/faiss_bench_data/{name}/xtrain.npy")
    print(name, "xb", xb.shape, "xq", xq.shape, "xtrain", xtrain.shape)