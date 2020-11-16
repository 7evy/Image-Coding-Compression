from chef import *

# Here are testing codes for all functions and classes in chef.py
# Variables to set are at the beginning of each part (##)


## rand_vector, estimate_P, entropy

length = 1000
X = rand_vector([0, 1], length)
print("Random "+str(length)+"-bits vector entropy :", entropy(X))


## draw_h

steps = 100
length = 1000
setX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
draw_h(steps, length, setX)


## estimate_M

IMAGE_RELATIVE_PATH = ""

# Loads an image from IMAGE_RELATIVE_PATH and converts it to a binary 2D array
Img = img.imread(IMAGE_RELATIVE_PATH)
ImgBW = []
for k in Img :
    ImgBW.append([])
    for q in k :
        if int(q[0]) == 1 :
            ImgBW[-1].append(1)
        else :
            ImgBW[-1].append(0)

p00, p10, p01, p11 = estimate_M(ImgBW)
ImgBWFlat = np.array(ImgBW).flatten()
[p0, p1] = estimate_P(ImgBWFlat)
H = -p0 *(p00*np.log2(p00) + p01*np.log2(p01)) -p1 *(p10*np.log2(p10) + p11*np.log2(p11))
print("Entropy with first order memory :", H)
print("Memoryless entropy :", entropy(ImgBWFlat))


## run_length_encoder

# print(run_length_encoder(I, "BW")) # Caution : terminal may crash !
print("Length of RLE code for", IMAGE_RELATIVE_PATH, ":", code_length(run_length_encoder(ImgBW)))


## k_means, image_from_kmeans

IMAGE_RELATIVE_PATH = ""

# Loads and flatten an RGB image for KMeans
Img2D = img.imread(IMAGE_RELATIVE_PATH)
_, n_cols, _ = np.shape(Img2D)
ImgFlat = []
for line in Img2D :
    for pixel in line :
        ImgFlat.append(pixel)

# KMeans with ten clusters
labels, clusters = k_means(ImgFlat, 10)
ImgLabels, ImgClustersFlat = images_from_kmeans(labels, clusters, n_cols)
print("Distortion with 10 clusters for", IMAGE_RELATIVE_PATH, ":", mean_squared_error(ImgFlat, ImgClustersFlat))
print("Coding rate with 10 clusters for", IMAGE_RELATIVE_PATH, ":", code_length(run_length_encoder(ImgLabels)) / code_length(ImgFlat))
plt.imshow(np.array([[int(rgb) for rgb in pixel] for pixel in ImgClustersFlat]).reshape(-1, n_cols, 3))
plt.axis("off")
plt.show()

# Distortion as a function of coding rate, with 2 to 20 clusters
Distortions = []
Coding_rates = []
for k in range(2, 21):
    labels, clusters = k_means(ImgFlat, k)
    ImgLabels, ImgClustersFlat = images_from_kmeans(labels, clusters, n_cols)
    Distortions.append(mean_squared_error(ImgFlat, ImgClustersFlat))
    Coding_rates.append(code_length(run_length_encoder(ImgLabels))/code_length(ImgFlat))
plt.plot(Coding_rates, Distortions)
plt.xlabel("Coding rate")
plt.ylabel("Distortion as MSE")
plt.show()


## estimate_joint_P, mutual_info_quantity, NMI, cross_NMI

# Uses the same image as KMeans, compares RGB channels
RGB_set = [k for k in range(256)]
ImgRed = [int(ImgFlat[k][0]) for k in range(len(ImgFlat))]
ImgBlue = [int(ImgFlat[k][2]) for k in range(len(ImgFlat))]
ImgGreen = [int(ImgFlat[k][1]) for k in range(len(ImgFlat))]
print(NMI(ImgRed, ImgBlue, RGB_set, RGB_set))
print(NMI(ImgBlue, ImgGreen, RGB_set, RGB_set))
print(NMI(ImgRed, ImgGreen, RGB_set, RGB_set))

# Displays NMIs between KMeans RGB channels and original ones, with 2 to 20 clusters
x_axis = range(2, 21)
y_axis_red = []
y_axis_blue = []
y_axis_green = []
for k in range(2, 21):
    labels, clusters = k_means(ImgFlat, k)
    ImgLabels, _ = images_from_kmeans(labels, clusters, n_cols)
    cluster = [[int(c[0]), int(c[1]), int(c[2])] for c in clusters]
    ImgFlatRed, ImgFlatBlue, ImgFlatGreen = [], [], []
    for line in ImgLabels :
        for label in line :
            ImgFlatRed.append(cluster[label][0])
            ImgFlatBlue.append(cluster[label][2])
            ImgFlatGreen.append(cluster[label][1])
    y_axis_red.append(NMI(ImgRed, ImgFlatRed, RGB_set, RGB_set))
    y_axis_blue.append(NMI(ImgBlue, ImgFlatBlue, RGB_set, RGB_set))
    y_axis_green.append(NMI(ImgGreen, ImgFlatGreen, RGB_set, RGB_set))
plt.plot(x_axis, y_axis_red, color="red")
plt.plot(x_axis, y_axis_blue, color="blue")
plt.plot(x_axis, y_axis_green, color="green")
plt.ylabel("NMI between original image and kmeans result")
plt.xlabel("Number of clusters")
plt.show()

# Compares DNA sequences (A, T, G, C, -) registered in a .mat file
DATA_RELATIVE_PATH = ""

# Loads the .mat file and formats it
DNA = loadmat(DATA_RELATIVE_PATH)
names = [k[0][0] for k in DNA.get("label")]
species = dict([names[k], k] for k in range(len(names)))
DNA_set = [0, 1, 2, 3, 4]
dna_conversion = dict((('-', 0), ('A', 1), ('C', 2), ('G', 3), ('T', 4)))
samples = [[dna_conversion.get(i[j]) for j in range(len(i))] for i in DNA.get("str")]

# Displays NMIs between all pairs of DNA samples with a heatmap
nmis = cross_NMI(names, samples)
sns.heatmap(nmis, annot=True, fmt=".3f", xticklabels=names, yticklabels=names)
plt.show()

# Prints highest NMI with human DNA
human_nmis = nmis[species.get("Human")]
human_nmis[species.get("Human")] = 0
m = max(human_nmis)
print("Highest NMI with human DNA sample is :")
print(names[human_nmis.index(m)], ":", m)


## mRMR_feature_selection, kNN_Classifier

DATA_RELATIVE_PATH = ""
FEATURE_NUMBER = 1
NEIGHBOR_NUMBER = 5

# Loads .mat file and formats it
mat = loadmat(DATA_RELATIVE_PATH)
Img2D = mat.get("data").tolist()
Img = []
for c in Img2D :
    for x in c :
        Img.append(x)
Img = np.asarray_chkfinite(Img).T.tolist()
Y = mat.get("label").flatten().tolist()

# Selects FEATURE_NUMBER features from the loaded data
FI, _ = mRMR_feature_selection(Img, Y, FEATURE_NUMBER)
print(FI)

# Accuracy of the kNN classification as a function of the number of selected features
Y_train, Y_test = train_test_split(Y, test_size=0.5, random_state=69)
x_axis = range(1, FEATURE_NUMBER+1)
y_axis = []
for k in range(1, FEATURE_NUMBER+1):
    mRMR_Img = [[pixel[j] for j in FI[:k]] for pixel in Img]
    X_train, X_test = train_test_split(mRMR_Img, test_size=0.5, random_state=69)
    classifier = kNN_Classifier(NEIGHBOR_NUMBER, X_train, Y_train)
    y_axis.append(classifier.overall_accuracy(X_test, Y_test))
plt.plot(x_axis, y_axis)
plt.xlabel("Number of features selected")
plt.ylabel("Overall accuracy of the kNN classifier")
plt.show()

# Shows no difference with randomly ordered features
FI = shuffle(FI)
mRMR_Img = [[pixel[j] for j in FI] for pixel in Img]
X_train, X_test = train_test_split(mRMR_Img, test_size=0.5, random_state=69)
classifier = kNN_Classifier(NEIGHBOR_NUMBER, X_train, Y_train)
print(FEATURE_NUMBER, "bands selected, ordered randomly :", classifier.overall_accuracy(X_test, Y_test))
