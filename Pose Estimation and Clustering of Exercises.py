import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
from numpy import linalg as LA
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

## Reading the activity pose data
df = pd.read_csv("60Activities-poseData.csv")

for i in df.columns.tolist()[1:]:
    splitval = [re.split(', |, |/[ |]', x.strip("[]")) for x in df[i].tolist()]
    convertval= [[float(t) for t in l] for l in splitval]
    df[i]  = convertval

#Gif creation
def gif(e):
    fig = plt.figure()
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax  = fig.add_subplot(111, projection ="3d")
    
    deg_l_knee = []
    deg_r_knee = []
    deg_l_hip = []
    deg_r_hip = []
    deg_r_shoulder = []
    deg_l_shoulder = []
    deg_l_elbow = []
    deg_r_elbow = []
    

    def plotting(i):      
        ax.clear()        
        ax.plot( 
            [-df['xX_left_wrist'][e][i], -df['xX_left_elbow'][e][i]], 
            [-df['yY_left_wrist'][e][i], -df['yY_left_elbow'][e][i]], 
            [-df['zZ_left_wrist'][e][i], -df['zZ_left_elbow'][e][i]], marker='o', color='red', linestyle='solid')
            
        ax.plot(
            
            [-df['xX_right_elbow'][e][i], -df['xX_right_shoulder'][e][i]], 
            [-df['yY_right_elbow'][e][i], -df['yY_right_shoulder'][e][i]], 
            [-df['zZ_right_elbow'][e][i], -df['zZ_right_shoulder'][e][i]], marker='o',color='red', linestyle='solid' )
        
    
        ax.plot(
            
            [-df['xX_left_shoulder'][e][i], -df['xX_right_shoulder'][e][i]], 
            [-df['yY_left_shoulder'][e][i], -df['yY_right_shoulder'][e][i]], 
            [-df['zZ_left_shoulder'][e][i], -df['zZ_right_shoulder'][e][i]], marker='o', color='red', linestyle='solid' )
     
        ax.plot(
            
            [-df['xX_right_shoulder'][e][i], -df['xX_right_hip'][e][i]], 
            [-df['yY_right_shoulder'][e][i], -df['yY_right_hip'][e][i]], 
            [-df['zZ_right_shoulder'][e][i], -df['zZ_right_hip'][e][i]], marker='o', color='red', linestyle='solid' )
                   
    
        ax.plot(
            
            [-df['xX_left_shoulder'][e][i], -df['xX_left_hip'][e][i]], 
            [-df['yY_left_shoulder'][e][i], -df['yY_left_hip'][e][i]], 
            [-df['zZ_left_shoulder'][e][i], -df['zZ_left_hip'][e][i]], marker='o', color='red', linestyle='solid' )
    
        ax.plot(
            
            [-df['xX_left_elbow'][e][i], -df['xX_left_shoulder'][e][i]], 
            [-df['yY_left_elbow'][e][i], -df['yY_left_shoulder'][e][i]], 
            [-df['zZ_left_elbow'][e][i], -df['zZ_left_shoulder'][e][i]], marker='o',color='red', linestyle='solid' )
        
        ax.plot(
    
            [-df['xX_right_wrist'][e][i], -df['xX_right_elbow'][e][i]], 
            [-df['yY_right_wrist'][e][i], -df['yY_right_elbow'][e][i]], 
            [-df['zZ_right_wrist'][e][i], -df['zZ_right_elbow'][e][i]], marker='o', color='red', linestyle='solid')
        
            
        ax.plot(
            [-df['xX_left_hip'][e][i], -df['xX_right_hip'][e][i]], 
            [-df['yY_left_hip'][e][i], -df['yY_right_hip'][e][i]], 
            [-df['zZ_left_hip'][e][i], -df['zZ_right_hip'][e][i]], marker='o',color='red', linestyle='solid' )
         
        ax.plot(
            [-df['xX_left_hip'][e][i], -df['xX_left_knee'][e][i]], 
            [-df['yY_left_hip'][e][i], -df['yY_left_knee'][e][i]], 
            [-df['zZ_left_hip'][e][i], -df['zZ_left_knee'][e][i]], marker='o',color='red', linestyle='solid' )
     
        ax.plot(
            [-df['xX_right_knee'][e][i], -df['xX_right_ankle'][e][i]], 
            [-df['yY_right_knee'][e][i], -df['yY_right_ankle'][e][i]], 
            [-df['zZ_right_knee'][e][i], -df['zZ_right_ankle'][e][i]], marker='o',color='red', linestyle='solid' )

        ax.plot(
            [-df['xX_left_knee'][e][i], -df['xX_left_ankle'][e][i]], 
            [-df['yY_left_knee'][e][i], -df['yY_left_ankle'][e][i]], 
            [-df['zZ_left_knee'][e][i], -df['zZ_left_ankle'][e][i]], marker='o',color='red', linestyle='solid' )
    
        ax.plot(
            [-df['xX_right_hip'][e][i], -df['xX_right_knee'][e][i]], 
            [-df['yY_right_hip'][e][i], -df['yY_right_knee'][e][i]], 
            [-df['zZ_right_hip'][e][i], -df['zZ_right_knee'][e][i]], marker='o',color='red', linestyle='solid' )
   
       #VECTORS FOR EACH POINT(Limbs)
        l_ankle_l_knee = np.array([df['xX_left_knee'][e][i] - df['xX_left_ankle'][e][i], df['yY_left_knee'][e][i] - df['yY_left_ankle'][e][i], df['zZ_left_knee'][e][i] - df['zZ_left_ankle'][e][i]])
        r_ankle_r_knee = np.array([df['xX_right_knee'][e][i] - df['xX_right_ankle'][e][i] , df['yY_right_knee'][e][i] - df['yY_right_ankle'][e][i] , df['zZ_right_knee'][e][i] - df['zZ_right_ankle'][e][i]])
        l_knee_l_hip = np.array([df['xX_left_knee'][e][i] - df['xX_left_hip'][e][i] , df['yY_left_knee'][e][i] - df['yY_left_hip'][e][i] , df['zZ_left_knee'][e][i] - df['zZ_left_hip'][e][i]])
        r_knee_r_hip = np.array([df['xX_right_knee'][e][i] - df['xX_right_hip'][e][i] , df['yY_right_knee'][e][i] - df['yY_right_hip'][e][i], df['zZ_right_knee'][e][i] - df['zZ_right_hip'][e][i]])
        l_hip_l_shoulder = np.array([df['xX_left_hip'][e][i] - df['xX_left_shoulder'][e][i] , df['yY_left_hip'][e][i] - df['yY_left_shoulder'][e][i], df['zZ_left_hip'][e][i] - df['zZ_left_shoulder'][e][i]])
        r_hip_r_shoulder = np.array([df['xX_right_hip'][e][i] - df['xX_right_shoulder'][e][i], df['yY_right_hip'][e][i] - df['yY_right_shoulder'][e][i] , df['zZ_right_hip'][e][i] - df['zZ_right_shoulder'][e][i]])
        l_shoulder_l_elbow = np.array([df['xX_left_shoulder'][e][i] - df['xX_left_elbow'][e][i] , df['yY_left_shoulder'][e][i] - df['yY_left_elbow'][e][i] , df['zZ_left_shoulder'][e][i] - df['zZ_left_elbow'][e][i]])
        r_shoulder_r_elbow = np.array([df['xX_right_shoulder'][e][i] - df['xX_right_elbow'][e][i], df['yY_right_shoulder'][e][i] - df['yY_right_elbow'][e][i] , df['zZ_right_shoulder'][e][i] - df['zZ_right_elbow'][e][i]])
        l_elbow_l_wrist = np.array([df['xX_left_elbow'][e][i] - df['xX_left_wrist'][e][i] , df['yY_left_elbow'][e][i] - df['yY_left_wrist'][e][i] , df['zZ_left_elbow'][e][i] - df['zZ_left_wrist'][e][i]])
        r_elbow_r_wrist = np.array([df['xX_right_elbow'][e][i] - df['xX_right_wrist'][e][i], df['yY_right_elbow'][e][i] - df['yY_right_wrist'][e][i], df['zZ_right_elbow'][e][i] - df['zZ_right_wrist'][e][i]])
        
       
        #DEGREE BETWEEN ALL JOINTS USING ABOVE VECTORS
        deg_l_knee_val = np.rad2deg(np.arccos(np.clip((np.inner(l_ankle_l_knee,l_knee_l_hip)/LA.norm(l_knee_l_hip)/LA.norm(l_ankle_l_knee)),-1.0,1.0)))
        deg_l_knee.append(deg_l_knee_val)
        h = 'Left Knee degree='+ str(int(deg_l_knee_val))
        deg_r_knee_val = np.rad2deg(np.arccos(np.clip((np.inner(r_ankle_r_knee,r_knee_r_hip)/LA.norm(r_knee_r_hip)/LA.norm(r_ankle_r_knee)),-1.0,1.0)))
        deg_r_knee.append(deg_r_knee_val)
        i = 'Right Knee degree='+ str(int(deg_r_knee_val))
        deg_l_hip_val = np.rad2deg(np.arccos(np.clip((np.inner(l_knee_l_hip,l_hip_l_shoulder)/LA.norm(l_hip_l_shoulder)/LA.norm(l_knee_l_hip)),-1.0,1.0)))
        deg_l_hip.append(deg_l_hip_val)
        j = 'Left Hip degree='+ str(int(deg_l_hip_val))
        deg_r_hip_val = np.rad2deg(np.arccos(np.clip((np.inner(r_knee_r_hip,r_hip_r_shoulder)/LA.norm(r_hip_r_shoulder)/LA.norm(r_knee_r_hip)),-1.0,1.0)))
        deg_r_hip.append(deg_r_hip_val)
        k = 'Right Hip degree='+ str(int(deg_r_hip_val))
        deg_r_shoulder_val = np.rad2deg(np.arccos(np.clip((np.inner(r_hip_r_shoulder,r_shoulder_r_elbow)/LA.norm(r_shoulder_r_elbow)/LA.norm(r_hip_r_shoulder)),-1.0,1.0)))
        deg_r_shoulder.append(deg_r_shoulder_val)
        l = 'Right Shoulder degree='+ str(int(deg_r_shoulder_val))
        deg_l_shoulder_val = np.rad2deg(np.arccos(np.clip((np.inner(l_hip_l_shoulder,l_shoulder_l_elbow)/LA.norm(l_shoulder_l_elbow)/LA.norm(l_hip_l_shoulder)),-1.0,1.0)))
        deg_l_shoulder.append(deg_l_shoulder_val)
        m = 'Left Shoulder degree='+ str(int(deg_l_shoulder_val))
        deg_l_elbow_val = np.rad2deg(np.arccos(np.clip((np.inner(l_shoulder_l_elbow,l_elbow_l_wrist)/LA.norm(l_elbow_l_wrist)/LA.norm(l_shoulder_l_elbow)),-1.0,1.0)))
        deg_l_elbow.append(deg_l_elbow_val)
        n = 'Left Elbow degree='+ str(int(deg_l_elbow_val))
        deg_r_elbow_val = np.rad2deg(np.arccos(np.clip((np.inner(r_shoulder_r_elbow,r_elbow_r_wrist)/LA.norm(r_elbow_r_wrist)/LA.norm(r_shoulder_r_elbow)),-1.0,1.0)))
        deg_r_elbow.append(deg_r_elbow_val)
        o = 'Right Elbow degree='+ str(int(deg_r_elbow_val))

        ax.view_init(-100, 80)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()
        ax.text2D(0.05, 0.95, h, transform=ax.transAxes,fontsize = 12, color = 'blue')
        ax.text2D(0.05, 0.92, i, transform=ax.transAxes,fontsize = 12, color = 'blue')
        ax.text2D(0.05, 0.89, j, transform=ax.transAxes,fontsize = 12, color = 'blue')
        ax.text2D(0.05, 0.86, k, transform=ax.transAxes,fontsize = 12, color = 'blue')
        ax.text2D(0.50, 0.95, l, transform=ax.transAxes,fontsize = 12, color = 'blue')
        ax.text2D(0.50, 0.92, m, transform=ax.transAxes,fontsize = 12, color = 'blue')
        ax.text2D(0.50, 0.89, n, transform=ax.transAxes,fontsize = 12, color = 'blue')
        ax.text2D(0.50, 0.86, o, transform=ax.transAxes,fontsize = 12, color = 'blue')      
        
        plt.close(fig)
    val = len(df['xX_left_wrist'][e])
    frames = val
    ani = animation.FuncAnimation(fig, plotting, frames=frames, interval=100)   
    ani.save(df["name"][e] + '.gif', writer='pillow')
    dataf = pd.DataFrame({'name': [df["name"][e]], 'deg_l_knee': [deg_l_knee], 'deg_r_knee': [deg_r_knee], 'deg_l_hip': [deg_l_hip], 'deg_r_hip': [deg_r_hip], 'deg_r_shoulder': [deg_r_shoulder], 'deg_l_shoulder': [deg_l_shoulder], 'deg_l_elbow': [deg_l_elbow], 'deg_r_elbow': [deg_r_elbow]})                          
    return dataf

datafin = pd.DataFrame()                    
for i in range (0,60):
    dataf = gif(i)    
    datafin = datafin.append(dataf, ignore_index=True)
datafin.to_csv('DegreeDetails.csv')

def remove_outlier_IQR(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    df_final=df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)))]    
    clensed = df_final.values.tolist()
    return clensed

def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
def Q1_calc(df):
    Q1=df.quantile(0.25)
    return Q1

Final_values =  pd.DataFrame()
clusterdf = pd.DataFrame()
columnvalues = ['deg_l_knee', 'deg_r_knee','deg_l_hip','deg_r_hip', 'deg_r_shoulder','deg_l_shoulder', 'deg_l_elbow', 'deg_r_elbow']
for i in range (0,60):   
    Moving = []
    Not_moving = []
    angledchanges = []
    Angle_changes_ = defaultdict(list) 
    rangedata = []
   
    for column in datafin[['deg_l_knee', 'deg_r_knee','deg_l_hip','deg_r_hip', 'deg_r_shoulder','deg_l_shoulder', 'deg_l_elbow', 'deg_r_elbow']] :
        #print("inside for loop")
        L = datafin[column][i]
        listofanglechanges = [y - x for x,y in zip(L,L[1:])]
        col = column[4:]
        datafr = pd.DataFrame (listofanglechanges, columns = ['angles'])
        df_outlier_removed=remove_outlier_IQR(datafr.angles)       
        angledchanges.append(df_outlier_removed)
        
        if (col == 'l_knee'):                
            l_knee = np.ptp(df_outlier_removed)
            rangedata.append(l_knee)
        elif (col == 'r_knee'):                
            r_knee = np.ptp(df_outlier_removed)
            rangedata.append(r_knee)
        elif (col == 'l_hip'):                
            l_hip = np.ptp(df_outlier_removed)
            rangedata.append(l_hip)
        elif (col == 'r_hip'):                
            r_hip = np.ptp(df_outlier_removed)
            rangedata.append(r_hip)
        elif (col == 'r_shoulder'):                
            r_shoulder = np.ptp(df_outlier_removed)
            rangedata.append(r_shoulder)
        elif (col == 'l_shoulder'):                
            l_shoulder = np.ptp(df_outlier_removed)
            rangedata.append(l_shoulder) 
        elif (col == 'l_elbow'):                
            l_elbow = np.ptp(df_outlier_removed)
            rangedata.append(l_elbow)                    
        else:             
            r_elbow = np.ptp(df_outlier_removed)
            rangedata.append(r_elbow)
        

    normdat = NormalizeData(rangedata)
    normdatdf = pd.DataFrame (normdat, columns = ['normaliseddata'])
    df_normdatdf=Q1_calc(normdatdf.normaliseddata)    
                           
    for k in range(0,8):
        colname = columnvalues[k]
        if(normdat[k]>df_normdatdf):
            Moving.append(colname[4:])
            Angle_changes_[colname[4:]].append(angledchanges[k])
        else:
            Not_moving.append(colname[4:])
            
    angle = dict(Angle_changes_)
    
    fin_result = pd.DataFrame({'name': datafin['name'][i], 'Moving are': [Moving], 'Not Moving are': [Not_moving], 'Angle Changes are': [angle]})
    Final_values = Final_values.append(fin_result, ignore_index=True)
      
    clustervals = pd.DataFrame({'name': datafin['name'][i], 'l_knee': [l_knee] , 'r_knee': [r_knee], 'l_hip': [l_hip], 'r_hip': [r_hip], 'r_shoulder': [r_shoulder], 'l_shoulder': [l_shoulder], 'l_elbow': [l_elbow], 'r_elbow': [r_elbow]})
    clusterdf = clusterdf.append(clustervals, ignore_index=True)

Final_values.to_csv('Final_Output.csv')
clusterdf.to_csv('DatasetforClustering.csv')

###### Some experiments on data #####
def experiment(n):
    gif(n)
    pd.options.display.max_colwidth = 80
    print(Final_values.iloc[n])
    
experiment(8)

experiment(9)

####### CLUSTERING ###########
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA
import seaborn as sns
clusterdf = clusterdf.reset_index(drop = True)
clusterdf.head()

### Run PCA on the data and reduce the dimensions in pca_num_components dimensions
def PCA(dafr, method,n):
    pca_num_components = 2
    reduced_data = IncrementalPCA(n_components=pca_num_components).fit_transform(dafr)
    results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])
    palette=['green','orange','brown','dodgerblue','red', 'purple']
    sns.scatterplot(x="pca1", y="pca2", hue=dafr['Cluster'], palette = palette[0:n], data=results, legend='full')
    plt.title(method + ' Clustering with 2 dimensions')
    plt.savefig(method + "_cluster_method.png")
    return plt.show()
## Elbow Method to find number of clusters
wcss=[]
x = clusterdf.iloc[:,2:10]
for i in range(2,7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(2,7)
plt.plot(number_clusters,wcss)
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

###### K Means
kmeans = KMeans(5)
kmeans.fit(x)
identified_clusters = kmeans.fit_predict(x)
print(identified_clusters)
kmeans_df = clusterdf.copy()
kmeans_df['Cluster'] = kmeans.labels_
kmeans_df.head()
kmeans_df.to_csv('K_Means_Clustering_Result.csv')

kmeans_df['name'][kmeans_df['Cluster'] == 0]

kmeans_df['name'][kmeans_df['Cluster'] == 1]

kmeans_df['name'][kmeans_df['Cluster'] == 2]

kmeans_df['name'][kmeans_df['Cluster'] == 3]

kmeans_df['name'][kmeans_df['Cluster'] == 4]

kmeans_df = kmeans_df.iloc[:,2:11]
kmeans_df.head()

PCA(kmeans_df, 'K-means', 5)

### Calculating silhouette Coefficient to identify best cluster for K medoid
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_samples, silhouette_score
sw = []
for i in range(2, 11):
    kMedoids = KMedoids(n_clusters = i, random_state = 0)
    kMedoids.fit(x)
    y_kmed = kMedoids.fit_predict(x)
    silhouette_avg = silhouette_score(x, y_kmed)
    sw.append(silhouette_avg)

plt.plot(range(2, 11), sw)
plt.title('Silhoute Score')
plt.xlabel('Number of clusters')
plt.ylabel('SW')      #within cluster sum of squares
plt.show()

# K Medoids
kMedoids = KMedoids(n_clusters = 2, random_state = 0)
kMedoids.fit(x)
y_kmed = kMedoids.fit_predict(x)
y_kmed

kmedoid_df = clusterdf.copy()
kmedoid_df['Cluster'] = y_kmed
kmedoid_df.head()
kmedoid_df.to_csv('K_Medoid_Clustering_Result.csv')

kmedoid_df = kmedoid_df.iloc[:,2:]
kmedoid_df.head()

PCA(kmedoid_df, 'K-medoid',2)

### BIRCH
from sklearn.cluster import Birch
model = Birch(branching_factor = 50, n_clusters = None, threshold = 1.5)
model.fit(x)
pred = model.predict(x)
pred

birch_df = clusterdf.copy()
birch_df['Cluster'] = pred
birch_df.head()
birch_df.to_csv('Birch_Clustering_Result.csv')

birch_df = birch_df.iloc[:,2:]
birch_df.head()

#PCA(birch_df, 'Birch')
pca_num_components = 2
reduced_data = IncrementalPCA(n_components=pca_num_components).fit_transform(birch_df)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])
sns.scatterplot(x="pca1", y="pca2", hue=birch_df['Cluster'], data=results)
plt.title('Birch Clustering with 2 dimensions')
plt.savefig("Birch_cluster_method.png")
plt.show()

######SLINK
import scipy
import scipy.cluster.hierarchy as sch     # 100 2-dimensional observations
d = sch.distance.pdist(x)   # vector of (100 choose 2) pairwise distances
L = sch.linkage(d, method='complete')
predsch = sch.fcluster(L, 0.5*d.max(), 'distance')
predsch

slink_df = clusterdf.copy()
slink_df['Cluster'] = predsch
slink_df.head()
slink_df.to_csv('Slink_Clustering_Result.csv')

slink_df = slink_df.iloc[:,2:]
slink_df.head()

PCA(slink_df, 'Slink',4)

#### soft clustering - Fuzzy C-means
from fcmeans import FCM
y = clusterdf.iloc[:,2:10]
fcm = FCM(n_clusters=5)
fcm.fit(y.to_numpy())
# outputs
fcm_centers = fcm.centers
fcm_labels = fcm.predict(y.to_numpy())

fcm_labels

fcmeans_df = clusterdf.copy()
fcmeans_df['Cluster'] = fcm_labels
fcmeans_df.head()
fcmeans_df.to_csv('FCMeans_Clustering_Result.csv')

fcmeans_df = fcmeans_df.iloc[:,2:]
fcmeans_df.head()

PCA(fcmeans_df , 'Fuzzy C-mean',5)

##### OPTICS
from sklearn.cluster import OPTICS
optics_clustering = OPTICS(min_samples=2, cluster_method= 'xi').fit_predict(x.to_numpy())
optics_clustering

optics_df = clusterdf.copy()
optics_df['Cluster'] = optics_clustering
optics_df.head()
optics_df.to_csv('Optics_Clustering_Result.csv')

optics_df = optics_df.iloc[:,2:]
optics_df.head()

#PCA(optics_df, 'Optics')
pca_num_components = 2
reduced_data = IncrementalPCA(n_components=pca_num_components).fit_transform(optics_df)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])
sns.scatterplot(x="pca1", y="pca2", hue=optics_df['Cluster'], data=results)
plt.title('Optics Clustering with 2 dimensions')
plt.savefig("Optics_cluster_method.png")
plt.show()

###### DBSCAN
#dbscan_clustering = DBSCAN(eps=3, min_samples=10).fit_predict(x.to_numpy())
from sklearn.cluster import DBSCAN
dbscan_clustering = DBSCAN(eps=30, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None).fit_predict(x.to_numpy())
dbscan_clustering

dbscan_df = clusterdf.copy()
dbscan_df['Cluster'] = dbscan_clustering
dbscan_df.head()
dbscan_df.to_csv('DBScan_Clustering_Result.csv')

dbscan_df = dbscan_df.iloc[:,2:]
dbscan_df.head()

PCA(dbscan_df, 'DBSCAN',2)

