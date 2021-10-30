# Εισαγωγή απαραίτητων βιβλιοθηκών για την οπτικοποιήση και εισαγωγή των δεδομένων.
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mplot
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from math import pi
import time
import io


# with open('./covtype.info') as file:
#    print(file.read())


names_from_file = ["Elevation                               quantitative    meters                       Elevation in meters\
					Aspect                                  quantitative    azimuth                      Aspect in degrees azimuth\
					Slope                                   quantitative    degrees                      Slope in degrees\
					Horizontal_Distance_To_Hydrology        quantitative    meters                       Horz Dist to nearest surface water features\
					Vertical_Distance_To_Hydrology          quantitative    meters                       Vert Dist to nearest surface water features\
					Horizontal_Distance_To_Roadways         quantitative    meters                       Horz Dist to nearest roadway\
					Hillshade_9am                           quantitative    0 to 255 index               Hillshade index at 9am, summer solstice\
					Hillshade_Noon                          quantitative    0 to 255 index               Hillshade index at noon, summer soltice\
					Hillshade_3pm                           quantitative    0 to 255 index               Hillshade index at 3pm, summer solstice\
					Horizontal_Distance_To_Fire_Points      quantitative    meters                       Horz Dist to nearest wildfire ignition points\
					Wilderness_Area (4 binary columns)      qualitative     0 (absence) or 1 (presence)  Wilderness area designation\
					Soil_Type (40 binary columns)           qualitative     0 (absence) or 1 (presence)  Soil Type designation\
					Cover_Type (7 types)                    integer         1 to 7                       Forest Cover Type designation"]

target_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
			   'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2',
			   'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7',
			   'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
			   'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',
			   'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
			   'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type']

#print(len(target_names))

data = pd.read_csv('./covtype.data')
current_columns_names = list(data.columns)

mapper = {}
for i, name in enumerate(current_columns_names):
	mapper[name] = target_names[i]

data = data.rename(columns=mapper)
# print(data.head(5))
# print(data.info())
# print(data.describe())
# print(data.dtypes)
# print(data.isnull().sum())

# Ορίζω το X_train, X_test, y_train και y_test, σύμφωνα με την εκφώνηση της άσκησης.
X_train = data.iloc[:15120, :-1]
X_test = data.iloc[15120:, :-1]

y_train = data.iloc[:15120, -1]
y_test = data.iloc[15120:, -1]

labels= 'Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz'

# Συναρτήσεις για τα γραφήματα των μεταβλητών του dataset και τα υλοποιώ με αυτό το τρόπο για δκή μου ευκολία
def plot_heatmap():
	plt.figure(figsize=(16, 12))
	plt.title('Correlation of the Features')
	sns.heatmap(data.corr(), linewidth = 0.01, square = False, cmap = plt.cm.RdBu, linecolor = 'white', annot = True, annot_kws={"size":4})
	plt.show()
	return

def plot_per_soil_types():
	data_ = data.copy()
	soil_type = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4',
				'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
				'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
				'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
				'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
				'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
				'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
				'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
				'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
				'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
	tmpList = []
	for c in data_[soil_type]:
	 tmpList += [str(c)] * data_[c].value_counts()[1]

	soil = pd.Series(tmpList)
	data_['Soil_Types'] = soil.values

	plt.figure(figsize=(16, 8))
	sns.countplot(data=data_, x='Soil_Types', hue='Cover_Type')
	plt.title('Number of Observations by Cover Type')
	plt.xticks(rotation=90)
	plt.legend(loc='upper right', labels = labels)
	plt.show()
	return



def plot_pie_percentage():
	class_dist = data.groupby('Cover_Type').size()
	fig1, ax1 = plt.subplots()
	fig1.set_size_inches(15,10)
	ax1.pie(class_dist, labels=labels, autopct='%1.1f%%')
	ax1.axis('equal')
	plt.title('Percentages of Cover Types',fontsize=15)
	plt.legend(loc='upper right', labels = labels)
	plt.show()
	return


def plot_per_Wilderness_Area():
	data_ = data.copy()
	wilderness_areas = ['Wilderness_Area1', 'Wilderness_Area2',
						 'Wilderness_Area3', 'Wilderness_Area4' ]

	tmpList = []
	for c in data_[wilderness_areas]:
		tmpList += [str(c)] * data_[c].value_counts()[1]

	wild = pd.Series(tmpList)
	data_['Wilderness_Areas'] = wild.values

	plt.figure(figsize=(16, 8))
	sns.countplot(data=data_, x='Wilderness_Areas', hue='Cover_Type')
	plt.title('Number of Observations by Wilderness_Area')
	plt.xticks()
	plt.legend(loc='upper right', labels = labels)
	plt.show()

	return

def plot_radar_chart():
	feat = data.copy()
	feat['Cover_Type'] = data['Cover_Type']
	unique_Cover_type = feat['Cover_Type'].unique()
	feat = feat.sort_values(by=['Cover_Type'])

	#Bρισκω καθε δείγμα σε ποιο Cover_Type είναι και τα ταξινομώ με βάση αυτό στη σειρά
	covtype = 0
	covtypes = []
	for k in range(1, unique_Cover_type.max()+1):
	 for i in range(0, len(data.values)):
		 if feat['Cover_Type'][i] == k:
			 covtype = covtype +1
	 covtypes.append(covtype)

	#Ο κάθε τύπος Cover_Type με τις εγγραφές που έχει
	cov_type_1 = feat.iloc[0:covtypes[0]]
	cov_type_2 = feat.iloc[covtypes[0]:covtypes[1]]
	cov_type_3 = feat.iloc[covtypes[1]:covtypes[2]]
	cov_type_4 = feat.iloc[covtypes[2]:covtypes[3]]
	cov_type_5 = feat.iloc[covtypes[3]:covtypes[4]]
	cov_type_6 = feat.iloc[covtypes[4]:covtypes[5]]
	cov_type_7 = feat.iloc[covtypes[5]:covtypes[6]]

	#Βρίσκω το μέσο όρο κάθε Cover_Type για να το πλοτάρω στη συνέχεια
	avg1 = cov_type_1.sum().div(covtypes[0])
	avg2 = cov_type_2.sum().div(covtypes[1] - covtypes[0])
	avg3 = cov_type_3.sum().div(covtypes[2] - covtypes[1])
	avg4 = cov_type_4.sum().div(covtypes[3] - covtypes[2])
	avg5 = cov_type_5.sum().div(covtypes[4] - covtypes[3])
	avg6 = cov_type_6.sum().div(covtypes[5] - covtypes[4])
	avg7 = cov_type_7.sum().div(covtypes[6] - covtypes[5])

	#Σε κάθε feature, έχοντας βρει το μέσο όρο, το πολλαπλασιάζω ετσι ώστε να φανεί και στο radar plot
	df = pd.DataFrame({
	'group': ['Type1','Type2','Type3','Type4','Type5','Type6','Type7'],
	'Elevation': [avg1[0],avg2[0],avg3[0],avg4[0],avg5[0],avg6[0],avg7[0]],
	'Aspect': [avg1[1]*20,avg2[1]*20,avg3[1]*20,avg4[1]*20,avg5[1]*20,avg6[1]*20,avg7[1]*20],
	'Slope': [avg1[2]*150,avg2[2]*150,avg3[2]*150,avg4[2]*150,avg5[2]*150,avg6[2]*150,avg7[2]*150],
	'Horizontal_Distance_To_Hydrology': [avg1[3]*10,avg2[3]*10,avg3[3]*10,avg4[3]*10,avg5[3]*10,avg6[3]*10,avg7[3]*10],
	'Vertical_Distance_To_Hydrology': [avg1[4]*50,avg2[4]*50,avg3[4]*50,avg4[4]*50,avg5[4]*50,avg6[4]*50,avg7[4]*50],
	'Horizontal_Distance_To_Roadways': [avg1[5],avg2[5],avg3[5],avg4[5],avg5[5],avg6[5],avg7[5]],
	'Hillshade_9am': [avg1[6]*10,avg2[6]*10,avg3[6]*10,avg4[6]*10,avg5[6]*10,avg6[6]*10,avg7[6]*10],
	'Hillshade_Noon': [avg1[7]*10,avg2[7]*10,avg3[7]*10,avg4[7]*10,avg5[7]*10,avg6[7]*10,avg7[7]*10],
	'Hillshade_3pm': [avg1[8]*20,avg2[8]*20,avg3[8]*20,avg4[8]*20,avg5[8]*20,avg6[8]*20,avg7[8]*20],
	'Horizontal_Distance_To_Fire_Points': [avg1[9],avg2[9],avg3[9],avg4[9],avg5[9],avg6[9],avg7[9]]
	})


	plt.figure(figsize=(20,15))
	#O αριθμός των μεταβλητών που θέλω να πλοτάρω
	types=list(df)[1:]
	N = len(types)

	#Σε κάθε άξονα βρίσκω τη γωνία της κάθε μεταβλητής ως εξής:
	#(γράφημα / αριθμός της μεταβλητής)
	angles = [n / float(N) * 2 * pi for n in range(N)]
	angles += angles[:1]
	ax = plt.subplot(111, polar=True)
	ax.set_theta_offset(pi / 2)
	ax.set_theta_direction(-1)

	# Φτιάχνω για κάθε μεταβλητή τον άξονα της.
	plt.xticks(angles[:-1], types)

	# Eδω φτιαχνω τον άξονα y, για να ειναι σε μέγεθος έτσι ώστε να φαίνονται οι διαφορές μεταξύ των γραφημάτων
	ax.set_rlabel_position(0)
	plt.yticks([1000,2000,3000], ["1000","2000","3000"], color="black", size=5)
	plt.ylim(0,3700)

	plt.title('Radar Chart of Features',fontsize=15)
	sns.set_style('whitegrid')

	for i in range(0,7):
		values = df.loc[i].drop('group').values.flatten().tolist()
		values += values[:1]
		ax.plot(angles, values, linewidth=1, linestyle='solid', label="Type %d" %i)
		ax.fill(angles, values, 'b', alpha=0.2)

	plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), labels = labels)
	plt.show()
	return


plot_pie_percentage()
plot_heatmap()
plot_per_soil_types()
plot_per_Wilderness_Area()
plot_radar_chart()




##########     B  EΡΩTHMA     ###############################################

# Επαλήθευση των μεγεθών των Χ_train kai X_test
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Kάνω scale τα δεδομένα
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


####   Μοντέλο Λογιστικής Παλινδρόμησης Β Ερωτήματος #############
log_model = LogisticRegression(penalty = 'l2', solver = 'lbfgs', max_iter = 10000, tol = 1e-3, C = 1.0, verbose = 1)
start_train = time.time()
log_model.fit(X_train, y_train)
end_train = time.time()

start_test = time.time()
preds_train = log_model.predict(X_train)
accuracy_train = accuracy_score(y_train, preds_train)
precision_train, recall_train, fscore_train, support_train = precision_recall_fscore_support(y_train, preds_train, average='macro')

preds_test = log_model.predict(X_test)
accuracy_test = accuracy_score(y_test, preds_test)
precision_test, recall_test, fscore_test, support_test = precision_recall_fscore_support(y_test, preds_test, average='macro')

end_test = time.time()
print('-'*100)
print('-'*100)
print('Training info: ')
print('Accuracy: {}'.format(round(accuracy_train, 4)))
print('Precision: {}'.format(round(precision_train, 4)))
print('Recall: {}'.format(round(recall_train, 4)))
print('F-Score: {}'.format(round(fscore_train, 4)))
print('Training Time: {}s'.format(round(end_train-start_train, 3)))
print('-'*100)
print('Testing info: ')
print('Accuracy: {}'.format(round(accuracy_test, 4)))
print('Precision: {}'.format(round(precision_test, 4)))
print('Recall: {}'.format(round(recall_test, 4)))
print('F-Score: {}'.format(round(fscore_test, 4)))
print('Testing Time: {}s'.format(round(end_test-start_test, 3)))
print('-'*100)
print('-'*100)

#Δημιουργώ το αρχείο file_1, για να καταγράψω τα αποτελέσματα των μοντέλων
file_1 = io.open('PENALTY_Newton0-cg_Sag-Lbfgs_results.txt', 'a', encoding="utf-8")
file_2 = io.open('TOL_Newton0-cg_Sag-Lbfgs_results.txt', 'a', encoding="utf-8")
file_3 = io.open('MAX_ITERS_Newton0-cg_Sag-Lbfgs_results.txt', 'a', encoding="utf-8")

# Εξέταση των παραμέτρων tols, max_iters και penalty στις τιμές που έχω στις αντίστοιχες λίστες τους
tols = [1e-6, 1e-4, 1e-1]
max_iters = [10, 100, 1000, 5000]
penalties_1 = ['none', 'l2']

for tol in tols:
	log_model_1 = LogisticRegression(solver = 'newton-cg', C = 1.0, tol = tol, max_iter = 1000, verbose = 1, random_state = 42)
	start_train_1 = time.time()
	results_1 = log_model_1.fit(X_train, y_train)
	end_train_1 = time.time()

	start_test_1 = time.time()
	preds_train = log_model_1.predict(X_train)
	accuracy_train_1 = accuracy_score(y_train, preds_train)
	precision_train_1, recall_train_1, fscore_train_1, support_train_1 = precision_recall_fscore_support(y_train, preds_train, average='macro')

	preds_test = log_model_1.predict(X_test)
	accuracy_test_1 = accuracy_score(y_test, preds_test)
	precision_test_1, recall_test_1, fscore_test_1, support_test_1 = precision_recall_fscore_support(y_test, preds_test, average='macro')
	end_test = time.time()

	file_2.write('-'*100+'\n')
	file_2.write('-'*100+'\n')
	file_2.write('Moντέλο Λογιστικής Παλινδρόμησης με solver: newton-cg, tol: '+str(tol)+',  και max_iter: 1000'+'\n')
	file_2.write('Training info: '+'\n')
	file_2.write('Accuracy: {}'.format(round(accuracy_train_1, 4))+'\n')
	file_2.write('Precision: {}'.format(round(precision_train_1, 4))+'\n')
	file_2.write('Recall: {}'.format(round(recall_train_1, 4))+'\n')
	file_2.write('F-Score: {}'.format(round(fscore_train_1, 4))+'\n')
	file_2.write('Training Time: {}s'.format(round(end_train_1-start_train_1, 4))+'\n')
	file_2.write('-'*100+'\n')
	file_2.write('Testing info: '+'\n')
	file_2.write('Accuracy: {}'.format(round(accuracy_test_1, 4))+'\n')
	file_2.write('Precision: {}'.format(round(precision_test_1, 4))+'\n')
	file_2.write('Recall: {}'.format(round(recall_test_1, 4))+'\n')
	file_2.write('F-Score: {}'.format(round(fscore_test_1, 4))+'\n')
	file_2.write('Testing Time: {}s'.format(round(end_test-start_test_1, 4))+'\n')
	file_2.write('-'*100+'\n')
	file_2.write('-'*100+'\n')

	log_model_2 = LogisticRegression(solver = 'sag', C = 1.0, tol = 0.01, max_iter = 1000, verbose = 1, random_state = 42)
	start_train_2 = time.time()
	results_2 = log_model_2.fit(X_train, y_train)
	end_train_2 = time.time()

	start_test_2 = time.time()
	preds_train = log_model_2.predict(X_train)
	accuracy_train_2 = accuracy_score(y_train, preds_train)
	precision_train_2, recall_train_2, fscore_train_2, support_train_2 = precision_recall_fscore_support(y_train, preds_train, average='macro')

	preds_test = log_model_2.predict(X_test)
	accuracy_test_2 = accuracy_score(y_test, preds_test)
	precision_test_2, recall_test_2, fscore_test_2, support_test_2 = precision_recall_fscore_support(y_test, preds_test, average='macro')
	end_test = time.time()

	file_2.write('-'*100+'\n')
	file_2.write('-'*100+'\n')
	file_2.write('Moντέλο Λογιστικής Παλινδρόμησης με solver: sag, tol: '+str(tol)+',  και max_iter: 1000'+'\n')
	file_2.write('Training info: '+'\n')
	file_2.write('Accuracy: {}'.format(round(accuracy_train_2, 4))+'\n')
	file_2.write('Precision: {}'.format(round(precision_train_2, 4))+'\n')
	file_2.write('Recall: {}'.format(round(recall_train_2, 4))+'\n')
	file_2.write('F-Score: {}'.format(round(fscore_train_2, 4))+'\n')
	file_2.write('Training Time: {}s'.format(round(end_train_2-start_train_2, 4))+'\n')
	file_2.write('-'*100+'\n')
	file_2.write('Testing info: '+'\n')
	file_2.write('Accuracy: {}'.format(round(accuracy_test_2, 4))+'\n')
	file_2.write('Precision: {}'.format(round(precision_test_2, 4))+'\n')
	file_2.write('Recall: {}'.format(round(recall_test_2, 4))+'\n')
	file_2.write('F-Score: {}'.format(round(fscore_test_2, 4))+'\n')
	file_2.write('Testing Time: {}s'.format(round(end_test-start_test_2, 4))+'\n')
	file_2.write('-'*100+'\n')
	file_2.write('-'*100+'\n')

	log_model_3 = LogisticRegression(solver = 'lbfgs', C = 1.0, tol = 0.01, max_iter = 1000, verbose = 1, random_state = 42)
	start_train_3 = time.time()
	results_3 = log_model_3.fit(X_train, y_train)
	end_train_3 = time.time()

	start_test_3 = time.time()
	preds_train = log_model_3.predict(X_train)
	accuracy_train_3 = accuracy_score(y_train, preds_train)
	precision_train_3, recall_train_3, fscore_train_3, support_train_3 = precision_recall_fscore_support(y_train, preds_train, average='macro')

	preds_test = log_model_3.predict(X_test)
	accuracy_test_3 = accuracy_score(y_test, preds_test)
	precision_test_3, recall_test_3, fscore_test_3, support_test_3 = precision_recall_fscore_support(y_test, preds_test, average='macro')
	end_test = time.time()

	file_2.write('-'*100+'\n')
	file_2.write('-'*100+'\n')
	file_2.write('Moντέλο Λογιστικής Παλινδρόμησης με solver: lbfgs, tol: '+str(tol)+',  και max_iter: 1000'+'\n')
	file_2.write('Training info: '+'\n')
	file_2.write('Accuracy: {}'.format(round(accuracy_train_3, 4))+'\n')
	file_2.write('Precision: {}'.format(round(precision_train_3, 4))+'\n')
	file_2.write('Recall: {}'.format(round(recall_train_3, 4))+'\n')
	file_2.write('F-Score: {}'.format(round(fscore_train_3, 4))+'\n')
	file_2.write('Training Time: {}s'.format(round(end_train_3-start_train_3, 4))+'\n')
	file_2.write('-'*100+'\n')
	file_2.write('Testing info: '+'\n')
	file_2.write('Accuracy: {}'.format(round(accuracy_test_3, 4))+'\n')
	file_2.write('Precision: {}'.format(round(precision_test_3, 4))+'\n')
	file_2.write('Recall: {}'.format(round(recall_test_3, 4))+'\n')
	file_2.write('F-Score: {}'.format(round(fscore_test_3, 4))+'\n')
	file_2.write('Testing Time: {}s'.format(round(end_test-start_test_3, 4))+'\n')
	file_2.write('-'*100+'\n')
	file_2.write('-'*100+'\n')

for max_iter in max_iters:
	log_model_1 = LogisticRegression(solver = 'newton-cg', C = 1.0, max_iter = max_iter, verbose = 1, random_state = 42)
	start_train_1 = time.time()
	results_1 = log_model_1.fit(X_train, y_train)
	end_train_1 = time.time()

	start_test_1 = time.time()
	preds_train = log_model_1.predict(X_train)
	accuracy_train_1 = accuracy_score(y_train, preds_train)
	precision_train_1, recall_train_1, fscore_train_1, support_train_1 = precision_recall_fscore_support(y_train, preds_train, average='macro')

	preds_test = log_model_1.predict(X_test)
	accuracy_test_1 = accuracy_score(y_test, preds_test)
	precision_test_1, recall_test_1, fscore_test_1, support_test_1 = precision_recall_fscore_support(y_test, preds_test, average='macro')
	end_test = time.time()

	file_3.write('-'*100+'\n')
	file_3.write('-'*100+'\n')
	file_3.write('Moντέλο Λογιστικής Παλινδρόμησης με solver: newton-cg,και max_iter: '+str(max_iter)+'\n')
	file_3.write('Training info: '+'\n')
	file_3.write('Accuracy: {}'.format(round(accuracy_train_1, 4))+'\n')
	file_3.write('Precision: {}'.format(round(precision_train_1, 4))+'\n')
	file_3.write('Recall: {}'.format(round(recall_train_1, 4))+'\n')
	file_3.write('F-Score: {}'.format(round(fscore_train_1, 4))+'\n')
	file_3.write('Training Time: {}s'.format(round(end_train_1-start_train_1, 4))+'\n')
	file_3.write('-'*100+'\n')
	file_3.write('Testing info: '+'\n')
	file_3.write('Accuracy: {}'.format(round(accuracy_test_1, 4))+'\n')
	file_3.write('Precision: {}'.format(round(precision_test_1, 4))+'\n')
	file_3.write('Recall: {}'.format(round(recall_test_1, 4))+'\n')
	file_3.write('F-Score: {}'.format(round(fscore_test_1, 4))+'\n')
	file_3.write('Testing Time: {}s'.format(round(end_test-start_test_1, 4))+'\n')
	file_3.write('-'*100+'\n')
	file_3.write('-'*100+'\n')

	log_model_2 = LogisticRegression(solver = 'sag', C = 1.0, max_iter = max_iter, verbose = 1, random_state = 42)
	start_train_2 = time.time()
	results_2 = log_model_2.fit(X_train, y_train)
	end_train_2 = time.time()

	start_test_2 = time.time()
	preds_train = log_model_2.predict(X_train)
	accuracy_train_2 = accuracy_score(y_train, preds_train)
	precision_train_2, recall_train_2, fscore_train_2, support_train_2 = precision_recall_fscore_support(y_train, preds_train, average='macro')

	preds_test = log_model_2.predict(X_test)
	accuracy_test_2 = accuracy_score(y_test, preds_test)
	precision_test_2, recall_test_2, fscore_test_2, support_test_2 = precision_recall_fscore_support(y_test, preds_test, average='macro')
	end_test = time.time()

	file_3.write('-'*100+'\n')
	file_3.write('-'*100+'\n')
	file_3.write('Moντέλο Λογιστικής Παλινδρόμησης με solver: sag,και max_iter: '+str(max_iter)+'\n')
	file_3.write('Training info: '+'\n')
	file_3.write('Accuracy: {}'.format(round(accuracy_train_2, 4))+'\n')
	file_3.write('Precision: {}'.format(round(precision_train_2, 4))+'\n')
	file_3.write('Recall: {}'.format(round(recall_train_2, 4))+'\n')
	file_3.write('F-Score: {}'.format(round(fscore_train_2, 4))+'\n')
	file_3.write('Training Time: {}s'.format(round(end_train_2-start_train_2, 4))+'\n')
	file_3.write('-'*100+'\n')
	file_3.write('Testing info: '+'\n')
	file_3.write('Accuracy: {}'.format(round(accuracy_test_2, 4))+'\n')
	file_3.write('Precision: {}'.format(round(precision_test_2, 4))+'\n')
	file_3.write('Recall: {}'.format(round(recall_test_2, 4))+'\n')
	file_3.write('F-Score: {}'.format(round(fscore_test_2, 4))+'\n')
	file_3.write('Testing Time: {}s'.format(round(end_test-start_test_2, 4))+'\n')
	file_3.write('-'*100+'\n')
	file_3.write('-'*100+'\n')

	log_model_3 = LogisticRegression(solver = 'lbfgs', C = 1.0, tol = 0.01, max_iter = 1000, verbose = 1, random_state = 42)
	start_train_3 = time.time()
	results_3 = log_model_3.fit(X_train, y_train)
	end_train_3 = time.time()

	start_test_3 = time.time()
	preds_train = log_model_3.predict(X_train)
	accuracy_train_3 = accuracy_score(y_train, preds_train)
	precision_train_3, recall_train_3, fscore_train_3, support_train_3 = precision_recall_fscore_support(y_train, preds_train, average='macro')

	preds_test = log_model_3.predict(X_test)
	accuracy_test_3 = accuracy_score(y_test, preds_test)
	precision_test_3, recall_test_3, fscore_test_3, support_test_3 = precision_recall_fscore_support(y_test, preds_test, average='macro')
	end_test = time.time()

	file_3.write('-'*100+'\n')
	file_3.write('-'*100+'\n')
	file_3.write('Moντέλο Λογιστικής Παλινδρόμησης με solver: lbfgs και max_iter: '+str(max_iter)+'\n')
	file_3.write('Training info: '+'\n')
	file_3.write('Accuracy: {}'.format(round(accuracy_train_3, 4))+'\n')
	file_3.write('Precision: {}'.format(round(precision_train_3, 4))+'\n')
	file_3.write('Recall: {}'.format(round(recall_train_3, 4))+'\n')
	file_3.write('F-Score: {}'.format(round(fscore_train_3, 4))+'\n')
	file_3.write('Training Time: {}s'.format(round(end_train_3-start_train_3, 4))+'\n')
	file_3.write('-'*100+'\n')
	file_3.write('Testing info: '+'\n')
	file_3.write('Accuracy: {}'.format(round(accuracy_test_3, 4))+'\n')
	file_3.write('Precision: {}'.format(round(precision_test_3, 4))+'\n')
	file_3.write('Recall: {}'.format(round(recall_test_3, 4))+'\n')
	file_3.write('F-Score: {}'.format(round(fscore_test_3, 4))+'\n')
	file_3.write('Testing Time: {}s'.format(round(end_test-start_test_3, 4))+'\n')
	file_3.write('-'*100+'\n')
	file_3.write('-'*100+'\n')


#Eξερεύνηση για την παράμετρο penalty
for penalty in penalties_1:
	log_model_1 = LogisticRegression(penalty = penalty, solver = 'newton-cg', C = 1.0, verbose = 1, random_state = 42)
	start_train_1 = time.time()
	results_1 = log_model_1.fit(X_train, y_train)
	end_train_1 = time.time()

	start_test_1 = time.time()
	preds_train = log_model_1.predict(X_train)
	accuracy_train_1 = accuracy_score(y_train, preds_train)
	precision_train_1, recall_train_1, fscore_train_1, support_train_1 = precision_recall_fscore_support(y_train, preds_train, average='macro')

	preds_test = log_model_1.predict(X_test)
	accuracy_test_1 = accuracy_score(y_test, preds_test)
	precision_test_1, recall_test_1, fscore_test_1, support_test_1 = precision_recall_fscore_support(y_test, preds_test, average='macro')
	end_test = time.time()

	file_1.write('-'*100+'\n')
	file_1.write('-'*100+'\n')
	file_1.write('Moντέλο Λογιστικής Παλινδρόμησης με solver: newton-cg και penalty: '+str(penalty)+'\n')
	file_1.write('Training info: '+'\n')
	file_1.write('Accuracy: {}'.format(round(accuracy_train_1, 4))+'\n')
	file_1.write('Precision: {}'.format(round(precision_train_1, 4))+'\n')
	file_1.write('Recall: {}'.format(round(recall_train_1, 4))+'\n')
	file_1.write('F-Score: {}'.format(round(fscore_train_1, 4))+'\n')
	file_1.write('Training Time: {}s'.format(round(end_train_1-start_train_1, 4))+'\n')
	file_1.write('-'*100+'\n')
	file_1.write('Testing info: '+'\n')
	file_1.write('Accuracy: {}'.format(round(accuracy_test_1, 4))+'\n')
	file_1.write('Precision: {}'.format(round(precision_test_1, 4))+'\n')
	file_1.write('Recall: {}'.format(round(recall_test_1, 4))+'\n')
	file_1.write('F-Score: {}'.format(round(fscore_test_1, 4))+'\n')
	file_1.write('Testing Time: {}s'.format(round(end_test-start_test_1, 4))+'\n')
	file_1.write('-'*100+'\n')
	file_1.write('-'*100+'\n')

	log_model_2 = LogisticRegression(penalty = penalty, solver = 'sag', verbose = 1, random_state = 42)
	start_train_2 = time.time()
	results_2 = log_model_2.fit(X_train, y_train)
	end_train_2 = time.time()

	start_test_2 = time.time()
	preds_train = log_model_2.predict(X_train)
	accuracy_train_2 = accuracy_score(y_train, preds_train)
	precision_train_2, recall_train_2, fscore_train_2, support_train_2 = precision_recall_fscore_support(y_train, preds_train, average='macro')

	preds_test = log_model_2.predict(X_test)
	accuracy_test_2 = accuracy_score(y_test, preds_test)
	precision_test_2, recall_test_2, fscore_test_2, support_test_2 = precision_recall_fscore_support(y_test, preds_test, average='macro')
	end_test = time.time()

	file_1.write('-'*100+'\n')
	file_1.write('-'*100+'\n')
	file_1.write('Moντέλο Λογιστικής Παλινδρόμησης με solver: sag και penalty: '+str(penalty)+'\n')
	file_1.write('Training info: '+'\n')
	file_1.write('Accuracy: {}'.format(round(accuracy_train_2, 4))+'\n')
	file_1.write('Precision: {}'.format(round(precision_train_2, 4))+'\n')
	file_1.write('Recall: {}'.format(round(recall_train_2, 4))+'\n')
	file_1.write('F-Score: {}'.format(round(fscore_train_2, 4))+'\n')
	file_1.write('Training Time: {}s'.format(round(end_train_2-start_train_2, 4))+'\n')
	file_1.write('-'*100+'\n')
	file_1.write('Testing info: '+'\n')
	file_1.write('Accuracy: {}'.format(round(accuracy_test_2, 4))+'\n')
	file_1.write('Precision: {}'.format(round(precision_test_2, 4))+'\n')
	file_1.write('Recall: {}'.format(round(recall_test_2, 4))+'\n')
	file_1.write('F-Score: {}'.format(round(fscore_test_2, 4))+'\n')
	file_1.write('Testing Time: {}s'.format(round(end_test-start_test_2, 4))+'\n')
	file_1.write('-'*100+'\n')
	file_1.write('-'*100+'\n')

	log_model_3 = LogisticRegression(penalty = penalty, solver = 'lbfgs', C = 1.0, tol = 0.01, max_iter = 1000, verbose = 1, random_state = 42)
	start_train_3 = time.time()
	results_3 = log_model_3.fit(X_train, y_train)
	end_train_3 = time.time()

	start_test_3 = time.time()
	preds_train = log_model_3.predict(X_train)
	accuracy_train_3 = accuracy_score(y_train, preds_train)
	precision_train_3, recall_train_3, fscore_train_3, support_train_3 = precision_recall_fscore_support(y_train, preds_train, average='macro')

	preds_test = log_model_3.predict(X_test)
	accuracy_test_3 = accuracy_score(y_test, preds_test)
	precision_test_3, recall_test_3, fscore_test_3, support_test_3 = precision_recall_fscore_support(y_test, preds_test, average='macro')
	end_test = time.time()

	file_1.write('-'*100+'\n')
	file_1.write('-'*100+'\n')
	file_1.write('Moντέλο Λογιστικής Παλινδρόμησης με solver: lbfgs και penalty: ' +str(penalty)+'\n')
	file_1.write('Training info: '+'\n')
	file_1.write('Accuracy: {}'.format(round(accuracy_train_3, 4))+'\n')
	file_1.write('Precision: {}'.format(round(precision_train_3, 4))+'\n')
	file_1.write('Recall: {}'.format(round(recall_train_3, 4))+'\n')
	file_1.write('F-Score: {}'.format(round(fscore_train_3, 4))+'\n')
	file_1.write('Training Time: {}s'.format(round(end_train_3-start_train_3, 4))+'\n')
	file_1.write('-'*100+'\n')
	file_1.write('Testing info: '+'\n')
	file_1.write('Accuracy: {}'.format(round(accuracy_test_3, 4))+'\n')
	file_1.write('Precision: {}'.format(round(precision_test_3, 4))+'\n')
	file_1.write('Recall: {}'.format(round(recall_test_3, 4))+'\n')
	file_1.write('F-Score: {}'.format(round(fscore_test_3, 4))+'\n')
	file_1.write('Testing Time: {}s'.format(round(end_test-start_test_3, 4))+'\n')
	file_1.write('-'*100+'\n')
	file_1.write('-'*100+'\n')


#Όπως και πριν έτσι και εδώ υλοποιώ αρχείο για την καταγραφή του μοντέλου Λογιστικής Παλινδρόμησης με solver=saga
penalties_2 = ['l1', 'l2', 'none']
file_5 = io.open('Saga_results.txt', 'a', encoding="utf-8")
for tol in tols:
	for max_iter in max_iters:
		for penalty in penalties_2:
			log_model_2 = LogisticRegression(penalty = penalty, solver = 'saga', C = 1.0, tol = tol, max_iter = max_iter, verbose = 1, random_state = 42)
			start_train = time.time()
			results = log_model_2.fit(X_train, y_train)
			end_train = time.time()

			start_test = time.time()
			preds_train = log_model_2.predict(X_train)
			accuracy_train = accuracy_score(y_train, preds_train)
			precision_train, recall_train, fscore_train, support_train = precision_recall_fscore_support(y_train, preds_train, average='macro')

			preds_test = log_model_2.predict(X_test)
			accuracy_test = accuracy_score(y_test, preds_test)
			precision_test, recall_test, fscore_test, support_test = precision_recall_fscore_support(y_test, preds_test, average='macro')
			end_test = time.time()

			file_5.write('-'*100+'\n')
			file_5.write('-'*100+'\n')
			file_5.write('Moντέλο Λογιστικής Παλινδρόμησης με solver: saga, penalty: '+str(penalty)+', tol: '+str(tol)+',  και max_iter: '+str(max_iter)+'\n')
			file_5.write('Training info: '+'\n')
			file_5.write('Accuracy: {}'.format(round(accuracy_train, 4))+'\n')
			file_5.write('Precision: {}'.format(round(precision_train, 4))+'\n')
			file_5.write('Recall: {}'.format(round(recall_train, 4))+'\n')
			file_5.write('F-Score: {}'.format(round(fscore_train, 4))+'\n')
			file_5.write('Training Time: {}s'.format(round(end_train-start_train, 4))+'\n')
			file_5.write('-'*100+'\n')
			file_5.write('Testing info: '+'\n')
			file_5.write('Accuracy: {}'.format(round(accuracy_test, 4))+'\n')
			file_5.write('Precision: {}'.format(round(precision_test, 4))+'\n')
			file_5.write('Recall: {}'.format(round(recall_test, 4))+'\n')
			file_5.write('F-Score: {}'.format(round(fscore_test, 4))+'\n')
			file_5.write('Testing Time: {}s'.format(round(end_test-start_test, 4))+'\n')
			file_5.write('-'*100+'\n')
			file_5.write('-'*100+'\n')

# Tώρα, υλοποιώ αρχείο για την καταγραφή του μοντέλου Λογιστικής Παλινδρόμησης με solver=liblinear

penalties_3 = ['l1', 'l2']
file = io.open('Liblinear_results.txt', 'a', encoding="utf-8")
for tol in tols:
	for max_iter in max_iters:
		for penalty in penalties_3:
			log_model_3 = LogisticRegression(penalty = penalty, solver = 'liblinear', C = 1.0, tol = tol, max_iter = max_iter, verbose = 1, random_state = 42)
			start_train = time.time()
			results = log_model_3.fit(X_train, y_train)
			end_train = time.time()

			start_test = time.time()
			preds_train = log_model_3.predict(X_train)
			accuracy_train = accuracy_score(y_train, preds_train)
			precision_train, recall_train, fscore_train, support_train = precision_recall_fscore_support(y_train, preds_train, average='macro')

			preds_test = log_model_3.predict(X_test)
			accuracy_test = accuracy_score(y_test, preds_test)
			precision_test, recall_test, fscore_test, support_test = precision_recall_fscore_support(y_test, preds_test, average='macro')
			end_test = time.time()

			file.write('-'*100+'\n')
			file.write('-'*100+'\n')
			file.write('Moντέλο Λογιστικής Παλινδρόμησης με solver: liblinear, penalty: '+str(penalty)+', tol: '+str(tol)+',  και max_iter: '+str(max_iter)+'\n')
			file.write('Training info: '+'\n')
			file.write('Accuracy: {}'.format(round(accuracy_train, 4))+'\n')
			file.write('Precision: {}'.format(round(precision_train, 4))+'\n')
			file.write('Recall: {}'.format(round(recall_train, 4))+'\n')
			file.write('F-Score: {}'.format(round(fscore_train, 4))+'\n')
			file.write('Training Time: {}s'.format(round(end_train-start_train, 4))+'\n')
			file.write('-'*100+'\n')
			file.write('Testing info: '+'\n')
			file.write('Accuracy: {}'.format(round(accuracy_test, 4))+'\n')
			file.write('Precision: {}'.format(round(precision_test, 4))+'\n')
			file.write('Recall: {}'.format(round(recall_test, 4))+'\n')
			file.write('F-Score: {}'.format(round(fscore_test, 4))+'\n')
			file.write('Testing Time: {}s'.format(round(end_test-start_test, 4))+'\n')
			file.write('-'*100+'\n')
			file.write('-'*100+'\n')


## Eξετάζω τη παράμετρο C στους πιθανούς solvers της Λογιστικής Παλινδρόμησης
C = [0.1, 10, 100]
solvers = ['newton-cg', 'lbfgs', 'liblinear','sag', 'saga']
file_6 = io.open('C.txt', 'a', encoding="utf-8")
for c in C:
	for solver in solvers:
		log_model_6 = LogisticRegression(C = c, solver = solver, max_iter = 1000, random_state = 42)
		start_train = time.time()
		results = log_model_6.fit(X_train, y_train)
		end_train = time.time()

		start_test = time.time()
		preds_train = log_model_6.predict(X_train)
		accuracy_train = accuracy_score(y_train, preds_train)
		precision_train, recall_train, fscore_train, support_train = precision_recall_fscore_support(y_train, preds_train, average='macro')

		preds_test = log_model_6.predict(X_test)
		accuracy_test = accuracy_score(y_test, preds_test)
		precision_test, recall_test, fscore_test, support_test = precision_recall_fscore_support(y_test, preds_test, average='macro')
		end_test = time.time()

		file_6.write('-'*100+'\n')
		file_6.write('-'*100+'\n')
		file_6.write('Moντέλο Λογιστικής Παλινδρόμησης με solver: '+str(solver)+' και C: '+str(c)+'\n')
		file_6.write('Training info: '+'\n')
		file_6.write('Accuracy: {}'.format(round(accuracy_train, 4))+'\n')
		file_6.write('Precision: {}'.format(round(precision_train, 4))+'\n')
		file_6.write('Recall: {}'.format(round(recall_train, 4))+'\n')
		file_6.write('F-Score: {}'.format(round(fscore_train, 4))+'\n')
		file_6.write('Training Time: {}s'.format(round(end_train-start_train, 4))+'\n')
		file_6.write('-'*100+'\n')
		file_6.write('Testing info: '+'\n')
		file_6.write('Accuracy: {}'.format(round(accuracy_test, 4))+'\n')
		file_6.write('Precision: {}'.format(round(precision_test, 4))+'\n')
		file_6.write('Recall: {}'.format(round(recall_test, 4))+'\n')
		file_6.write('F-Score: {}'.format(round(fscore_test, 4))+'\n')
		file_6.write('Testing Time: {}s'.format(round(end_test-start_test, 4))+'\n')
		file_6.write('-'*100+'\n')
		file_6.write('-'*100+'\n')

######   LDA Analysis  ######
##  Ακολουθεί  υλοποίηση του μοντέλου LDA

lda_model = LDA()
start_train_lda = time.time()
results = lda_model.fit(X_train, y_train)
end_train_lda = time.time()

start_test_lda = time.time()
preds_train = lda_model.predict(X_train)
accuracy_train = accuracy_score(y_train, preds_train)
precision_train, recall_train, fscore_train, support_train = precision_recall_fscore_support(y_train, preds_train, average='macro')

preds_test = lda_model.predict(X_test)
accuracy_test = accuracy_score(y_test, preds_test)
precision_test, recall_test, fscore_test, support_test = precision_recall_fscore_support(y_test, preds_test, average='macro')

end_test_lda = time.time()

print('-'*100)
print('-'*100)
print('Training info: ')
print('Accuracy: {}'.format(round(accuracy_train, 4)))
print('Precision: {}'.format(round(precision_train, 4)))
print('Recall: {}'.format(round(recall_train, 4)))
print('F-Score: {}'.format(round(fscore_train, 4)))
print('Training Time: {}s'.format(round((end_train_lda-start_train_lda), 4)))
print('-'*100)
print('Testing info: ')
print('Accuracy: {}'.format(round(accuracy_test, 4)))
print('Precision: {}'.format(round(precision_test, 4)))
print('Recall: {}'.format(round(recall_test, 4)))
print('F-Score: {}'.format(round(fscore_test, 4)))
print('Testing Time: {}s'.format(round((end_test_lda-start_test_lda), 4)))
print('-'*100)
print('-'*100)
