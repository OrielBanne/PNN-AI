import numpy as np
# do this before importing pylab or pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.vq import vq, kmeans
data = np.vstack((np.random.rand(200,2) + np.array([.5, .5]),np.random.rand(200,2)))
centroids2, _ = kmeans(data, 2)
idx2,_ = vq(data,centroids2)
# scatter plot without centroids
fig = plt.figure(1)
plt.plot(data[:,0],data[:,1], 'o')
fig.savefig('test1.png')
# scatter plot with 2 centroids
fig = plt.figure(2)
plt.plot(data[:,0],data[:,1],'o')
plt.plot(centroids2[:,0],centroids2[:,1],'sm',markersize=16) # scatter plot with 2 centroids and point colored by cluster
fig.savefig('test2.png')
fig = plt.figure(3)
plt.plot(data[idx2==0,0],data[idx2==0,1],'ob',data[idx2==1,0],data[idx2==1,1],'or')
plt.plot(centroids2[:,0],centroids2[:,1],'sm',markersize=16)
centroids3, _ = kmeans(data, 3)
idx3,_ = vq(data,centroids3)
fig.savefig('test3.png')
# scatter plot with 3 centroids and points colored by cluster
fig = plt.figure(4)
plt.plot(data[idx3==0,0],data[idx3==0,1],'ob',data[idx3==1,0],data[idx3==1,1],'or',data[idx3==2,0],data[idx3==2,1],'og')
plt.plot(centroids3[:,0],centroids3[:,1],'sm',markersize=16)
fig.savefig('test4.pdf')
# calling show() will open your plots in windows, each opening # when you close the previous one
# you can save rather than opening the plots using savefig() plt.show()
