<p align="center">
<img src="https://user-images.githubusercontent.com/80779522/139317029-30a94b56-aa6a-49ab-b080-2e44e67b08a9.jpg" width="500" />

  
# Forest-Coverage-Type
  
To develop strategies for ecosystem, we require descriptive knowledge of available forest lands. This descriptive information facilitates easy and perfect decision-making process for resource planners. The aim of this repository is to predict the type of vegetation that covers a forest area based on its characteristics through Logistic Regression technique and Linear Discriminant Analysis (LDA).
  
# Dataset
  
In this work, for classification of forest cover type, we have used dataset available on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/covertype). The dataset [```covtype.data```](https://raw.githubusercontent.com/georgios-kalomitsinis/Forest-Coverage-Type/master/covtype.data) file), is described in the file [```covtype.info```](https://github.com/georgios-kalomitsinis/Forest-Coverage-Type/blob/master/covtype.info). The first 15120 records are considered to be used for training and the rest for evaluation.

<div align="center">
  
| Forest type | Record count | Record percentage
| :---: | :---: | :---: | 
Spruce Fir | 211840 | 36.46
Lodgepole Pine | 283301 | 48.76 
Ponderosa Pine | 35754 | 6.15
Cottonwood/Willow | 2747 |0.47
Aspen | 9493 | 1.63
Douglas-fir | 17367 | 2.99
Krummholz | 20510 | 3.53
Total | 581012 | 100.00
<figcaption align = "center"><p align="center">Table 1. Forest cover type class distribution.</figcaption>
</figure>

Attribute name | Measurement unit | Attribute description
| :---: | :---: | :---: |
Elevation | Meters | Elevation
Aspect  |Azimuth  | Aspect in degrees
Slope  | Degrees | Slope
Horizontal distance to hydrology  |Meters  |Horizontal distance to nearest surface water features
Vertical distance to hydrology  |Meters  |Vertical distance to nearest surface water features
Horizontal distance to roadways  |Meters | Horizontal distance to nearest roadway
Hill shade at 9 am  |0–255 (Index)  |Hill shade index at 9am, summer solstice
Hill shade at noon  |0–255 (index) | Hill shade index at noon, summer solstice
Hill shade at 3 pm  |0–255 (Index) | Hill shade index at 3pm, summer solstice
Horizontal distance to fire points  |Meters  |Horizontal distance to nearest wildfire ignition points
Wilderness area (4 binary columns)  |0 (Absence) 1 (Presence)  |Wilderness area designation
Soil type (40 binary columns)  |0 (Absence) 1 (Presence) |Type of soil
Forest cover type (7 types) | 1–7  |Type of forest cover
</div>
<figcaption align = "center"><p align="center">Table 2. Data attribute information.</figcaption>
</figure>

The categorical variables are:

* Wilderness_Area (4 binary columns) qualitative 0 (absence) or 1 (presence)
* Soil_Type (40 binary columns) qualitative 0 (absence) or 1 (presence)
* Cover_Type (7 types) integer 1 to 7 Forest Cover Type designation


<p align="center">
<img src="https://user-images.githubusercontent.com/80779522/139328113-deb95ca2-8b89-4d81-89e1-b103f2aff180.png" width="650" />
<figcaption align = "center"><p align="center">Figure 1. Percentages of Cover Types.</figcaption>

## Modelling and Evaluation
**ALGORITHMS**

* *Logistic regression*
  * __LBFGS__ soliving algorith
  * 10<sup>-3</sup> __tolerance__
  * __L2__ normalization
  * weight __C__ equals to 1.0
* *Linear Discriminant Analysis* (LDA)
  
In Logistic Regression model, we define a “grid” of parameters that we would want to test out in the model and select the best model using ```GridSearchCV```, but no in LDA.

**METRICS**

<div align="center">

| Name | Formula | 
| :---: | :---: | 
| Accuracy | ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Cfrac%7BTP&plus;TN%7D%7BTP&plus;FP&plus;FN&plus;TN%7D) |
| Precision | ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Cfrac%7BTP%7D%7BTP&plus;FP%7D) |
| Recall | ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Cfrac%7BTP%7D%7BTP&plus;FN%7D) |
| F-Score | ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Cfrac%7B2%5Ctimes%20%28Recall%20%5Ctimes%20Precision%29%7D%7BRecall%20&plus;%20Precision%7D) |

</div>
<figcaption align = "center"><p align="center">Table 3. Calculated metrics where TP, TN, FP, FN corresponds to True Positives, True Negatives, False Negatives and False Positives, respectively.</figcaption>
</figure>

## Dependencies 
Install all the neccecary dependencies using ```pip3 install <package name>```
  
Required packages:
```
  - numpy (Version >= 1.19.4)
  - matplotlib (Version >= 3.4.3)
  - scikit-learn (Version >= 0.22.2)
  - seaborn (Version >= 0.10.1)
  - pandas (Version >= 1.0.3)
```

## License
This project is licensed under the [MIT License](https://github.com/georgios-kalomitsinis/Forest-Coverage-Type/blob/master/LICENSE).
