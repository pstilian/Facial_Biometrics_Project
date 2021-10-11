
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC


print(__doc__)

# 输出进度日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# 下载数据并加载为numpy数组。
'''lfw_people = fetch_lfw_people(min_faces_per_person=10,resize=0.4)'''
lfw_people = fetch_lfw_people(resize=0.4)
# 获得图像数组的形状(用于绘图)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# 预测的目标是人的ID
y = lfw_people.target
target_names = lfw_people.target_names  #所有class的名称
n_classes = target_names.shape[0]      #target_names数组的行数，即class的个数

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# Split into a training set and a test set using a stratified k fold

# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
# #############################################################################
# 在人脸数据集(当做无标记数据)上计算PCA (eigenfaces，特征脸): Principal Component Analysis
# 无监督特征提取 / 降维  150
n_components = 40

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',  #n_components表示主成分的方差和所占的最小比例阈值
          whiten=True).fit(X_train)                         #白化会去除变换信号中的一些信息(分量的相对方差尺度)
print("done in %0.3fs" % (time() - t0))                     #此处是在降维

eigenfaces = pca.components_.reshape((n_components, h, w))  #此处n_components=150。 h=50,w=37未发生变化

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)       #pca.transform应该也是降维，转换前size为（1673,1850）转换后为（1673,150）
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

# #############################################################################
# 训练SVM分类模型

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],   #param_grid 像function传递系数,应该就是后面对比不同系数找到Best estimator found by grid search
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
'''iid=False'''
clf = GridSearchCV(SVC(kernel='rbf'),param_grid, cv=5,iid=False)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:") #这里，用上面的系数找到最佳模型
print(clf.best_estimator_)

# #############################################################################
# 在测试集上评估模型的量化效果

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

#调用classification_report显示出预测结果
print(classification_report(y_test, y_pred, target_names=target_names))  
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

# #############################################################################
# 使用 matplotlib 定性分析预测结果

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# 绘制部分测试集的预测结果

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# 绘制几个最重要的特征脸的相册

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
