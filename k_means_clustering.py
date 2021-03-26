from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist # dùng để tính khoảng cách giữa các cặp điểm trong hai tập
# hợp một cách hiệu quả

b = np.random.seed(18)
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
print('X: \n', X)
print(X.shape)
K = 3 # 3 clusters
original_label = np.asarray([0]*N + [1]*N + [2]*N).T
print(np.amax(original_label))
print(original_label)
#hiển thị dữ liệu trên đồ thị
def kmeans_display(X, label):
	K = np.amax(label) + 1
	X0 = X[label == 0, :]
	X1 = X[label == 1, :]
	X2 = X[label == 2, :]

	plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8) # tam giac xanh duong
	plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8) # hinh tron xanh la
	plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8) # hinh vuong do

	plt.axis('equal')
	plt.plot()
	plt.show()

#khởi tạo các tâm cụm
def kmeans_init_centroids(X, k):
	#randomly pick k rows of X as initial centers
	return X[np.random.choice(X.shape[0], k, replace = False)] #chọn ngẫu nhiên giá trị trong ma trận X,
	#thông số replace giúp ngăn chặn các phần tử lặp lại

#tìm nhãn mới cho các điểm khi biết các tâm cụm (cố định M: tâm các cụm, tìm Y: nhãn các clusters)
def kmeans_assign_labels(X, centroids):
	#caculate pairwise distances btw datat and centers
	D = cdist(X, centroids)
	#return index  of the closest center
	return np.argmin(D, axis = 1)

# cập nhật các tâm cụm khi biết nhãn của từng điểm.
def kmeans_update_centroids(X, labels, K):
	centroids = np.zeros((K, X.shape[1])) # tạo ma trận zeros 3 hàng và 2 cột
	for k in range(K):
		#collect all points assigned to the k-th cluster
		Xk = X[labels == k, :]
		#take average
		centroids[k, :] = np.mean(Xk, axis = 0)
	return centroids

#kiểm tra điều kiện dừng của thuật toán
def has_converged(centroids, new_centroids):
	#return True if two sets of centers are the same
	return (set([tuple(a) for a in centroids]) == set([tuple(a) for a in new_centroids]))

# phần chính của phân cụm K-means:
def kmeans(X, K):
	centroids = [kmeans_init_centroids(X, K)]
	labels = []
	it = 0
	while True:
		labels.append(kmeans_assign_labels(X, centroids[-1]))
		new_centroids = kmeans_update_centroids(X, labels[-1], K)
		if has_converged(centroids[-1], new_centroids):
			break
		centroids.append(new_centroids)
		it += 1
	return(centroids, labels, it)

centroids, labels, it = kmeans(X, K)
print("Centers found by our algorithm:\n", centroids[-1])
kmeans_display(X, labels[-1])