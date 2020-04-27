---
layout : post
---

# Machine Learning Final Project --- <br>Grade Prediction 

:::info
Team15.zip 

-- [Team15_Palmistry_CNN.py (link)](https://colab.research.google.com/drive/1qOBXs8kU-LowhJ_fLjer1-WAwVsIapRs)-> Image classification using CNN 

-- [Team15_Model.py (link)](https://colab.research.google.com/drive/1TKEf1ixWZUbR3qNwHlyeTU5fJTdbYOQK) -> Classification using Random Forest and Logistic Regression

-- Team15.pptx -> presentation ppt

-- Team15.html -> Report


:::
*Group 15:
0616225 張承遠, 0616093 吳柏憲, 0612019 鄧遠翔, 0616227 陳亭妤, 0616228 莊于萱*

[Toc]


## Before reading:
:::danger
I do this project on CoLab due to free GPU environment. I  **do file manipulation**, **use command line** and **fetch data from my GoogleDrive** on the virtual machine in the environment. It may **cause some error** if executing the program in other environment.  
:::




## Introduction:
Try to predict Student's Grade with some specific features.
==features: 第幾類組、平均睡幾個小時、家境、性別、每天讀書時間 etc.==

Furthermore, try to predict Student's Grade with a palm image.


## Motivation:
Due to the strongness of Machine Learning, we tried to do something special that human is hard to solve. **Then, we came up with an idea -- Palmistry**.

We are curious about whether fortune teller is a superstition or is actually truthful. Hence, we collect palm image data and try to find the mysterious relation between palm and grade.
## Data_Collection:
Questionnaire posted on social media (Facebook, Instagram, Line etc.) with 17 questions. 
(16 specific feature and 1 palm image)

Finally, we got about 180 responses with some missing data. (some questions are answered optionally.)
## Data_Preprocessing:
- **Fill missing value:**
Fill missing value with highest frequency if feature is categorical.
Fill missing value with average if feature is numerical.
- **Deal with noise data:**
Some data are noisy. ex: 就讀學校：鬼殺隊
Some data are not in the same format. ex: 就讀學校：交通大學、交大、交通 etc.
**Ex:**
```python=
data_csv['生理性別'].fillna("男",inplace=True)
```
We manually modified each data to the same format and treated noisy data as missing value.
- **one-hot encoding and label encoding:**
For categorical data, we do label encoding and one-hot encoding to transform all feature to numerical type.
==**Code Demonstration:**==

    **Explanation:**
Split the feature into numerical, unused, and categorical.
Using function *get_dummies()* can easily do label encoding and one-hot encoding.
```python=
unused_feat = ['翹課堂數/上課堂數', '就讀的學校', '時間戳記', '左手的掌心照','喜歡的顏色']
numerical_feat = ['近視度數', '每週運動幾次', '如果有打工的話 打工時薪多少', '你喜歡自己讀的科系嗎', '平均每天讀書時間(小時)']
target_feat = '平均成績％數'


data_drop = data_csv.drop(columns= [target_feat] + unused_feat + numerical_feat)
data_encoding = pd.get_dummies(data_drop)

data_t = pd.concat([data_encoding, data_csv.loc[:, numerical_feat]], axis=1)
data_t
```
- **Image resizing, Image transforming:**
For image, we do 
  1. resizing to uniform the image size.
  2. transforming to reduce noise in image. (GrayScale, Gaussian Filter)
**==Code Demonstration:==**
```python=
from PIL import Image, ExifTags
%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import functools
from scipy.ndimage import gaussian_filter

def image_transpose_exif(im):
    """
        Apply Image.transpose to ensure 0th row of pixels is at the visual
        top of the image, and 0th column is the visual left-hand side.
        Return the original image if unable to determine the orientation.

        As per CIPA DC-008-2012, the orientation field contains an integer,
        1 through 8. Other values are reserved.
    """

    exif_orientation_tag = 0x0112
    exif_transpose_sequences = [                   # Val  0th row  0th col
        [],                                        #  0    (reserved)
        [],                                        #  1   top      left
        [Image.FLIP_LEFT_RIGHT],                   #  2   top      right
        [Image.ROTATE_180],                        #  3   bottom   right
        [Image.FLIP_TOP_BOTTOM],                   #  4   bottom   left
        [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],  #  5   left     top
        [Image.ROTATE_270],                        #  6   right    top
        [Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],  #  7   right    bottom
        [Image.ROTATE_90],                         #  8   left     bottom
    ]

    try:
        seq = exif_transpose_sequences[im._getexif()[exif_orientation_tag]]
    except Exception:
        return im
    else:
        return functools.reduce(type(im).transpose, seq, im)

def convertjpg(jpgfile,width=384,height=512):
    try :
        image = Image.open(jpgfile)
        
        image = image_transpose_exif(image) # rotate image (some picture will be rotated after resizing, need to rotate back)
        image = image.convert('LA') # convert to grayscale
        image=image.resize((width, height), Image.ANTIALIAS) # resize image

        os.remove(os.path.join(os.getcwd(), jpgfile))
        
        img_convert = np.array(image)
        img_c = gaussian_filter(img_convert, sigma=0.7) # Gaussian Filter
        image = Image.fromarray(img_c )
        
        image.save(jpgfile)
        return 1

    except Exception as e:
        print(e)
        return 0
```
**Implementation of dividing data and converting image:**
*For simplicity, we divide grade into two class: 0-40% and 41-100%. 
The reason we choose the segmented point is to check whether he can promote to graduated or not.*
**==Code Demonstration:==**

```python=
cnt1, cnt2, cnt3 = 1, 1, 1

for row in  data_csv.index:
    # grade = data_csv.loc[row, "生理性別"]
    grade = data_csv.loc[row, "平均成績％數"]
    pic_ID = data_csv.loc[row, "左手的掌心照"]
    
    pic_ID = pic_ID[pic_ID.find("id")+3:]
    if pic_ID in ["1L-DTFqfoj0MKAyG1MCIBYuC0tCTUNby1", "1saw4I-6_Oo-37tOcSqq5ceLA-WIz_vj4", "1invk7pqN5BeaXBtXa3gQRTZ7vICmm08F", "1UrImTDWGqBPHtP4fzREf77i7Kow7w23t", "1aPl6uKvUYwGL-yt6I1x8ILDVONdFJTeU", "1Cf9lkd89gRhDbJxl9POhPFPWy8uGVcgG", "1FwU9BEojTnnCZ_k2NwmUzukuyAl_LkB6", "1EmWAdyIoK7QZ7I9wkK1tSXiY6K4V1zHS" ]:
        continue
    # print(pic_ID)
    downloaded = drive.CreateFile({'id': pic_ID})
    print(pic_ID)

    if grade=="0-10%" or grade=="11-20%" or grade=="21-30%" or grade=="31-40%":
        pic_name = "first_" + str(cnt1) + ".png"
        print(pic_name)
        downloaded.GetContentFile(pic_name)
        succ = convertjpg(pic_name)
        if succ==0:
            os.remove(os.getcwd()+"/"+pic_name)
            continue;
        cnt1 += 1

        img=mpimg.imread(pic_name)
        imgplot = plt.imshow(img)
        plt.show()
        shutil.move(pic_name, "first/")
    
    else:
        pic_name = "second_" + str(cnt2) + ".png"
        print(pic_name)
        downloaded.GetContentFile(pic_name)
        succ = convertjpg(pic_name)
        if succ==0:
            os.remove(os.getcwd()+"/"+pic_name)
            continue;
        cnt2 += 1

        img=mpimg.imread(pic_name)
        imgplot = plt.imshow(img)
        plt.show()
        shutil.move(pic_name, "second/")
```
**Part of output:**
![](https://i.imgur.com/setV2Uy.png)

- **Train set, Validation set and Test set**
  For normal features classification, we divide data to training set(70%) and testing set(30%)
  **==Code Demonstration:==**
  ```python=
  from sklearn.model_selection import train_test_split

  X = data_t
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
  ```
  For image classification, we divide data to training set(50%), validation set(25%), testing set(25%)
  **==Code Demonstration:==**
```python=
## splits indices for a folder into train, validation, and test indices with random sampling
    ## input: folder path
    ## output: train, valid, and test indices    
def split_indices(folder,seed1,seed2):    
    n = len(os.listdir(folder))
        
    full_set = list(range(1,n+1))
    ## train indices
    ran.seed(seed1)
    train = ran.sample(list(range(1,n+1)),int(.5*n))

    ## temp
    remain = list(set(full_set)-set(train))
    # valid = list(set(full_set)-set(train))

    ## separate remaining into validation and test
    ran.seed(seed2)
    valid = ran.sample(remain,int(.5*len(remain)))
    test = list(set(remain)-set(valid))
    
    return(train,valid, test)

## gets file names for a particular type of trash, given indices
    ## input: waste category and indices
    ## output: file names 
def get_names(waste_type,indices):
    file_names = [waste_type+"_"+str(i)+".png" for i in indices]
    return(file_names)    

## moves group of source files to another folder
    ## input: list of source files and destination folder
    ## no output
def move_files(source_files,destination_folder):
    for file in source_files:
        shutil.copy(file,destination_folder)
        
#####################################################     
    
## paths will be train/cardboard, train/glass, etc...
subsets = ['train','valid']

grade_types = ['first', 'second']
!rm -r data/train data/valid data/test
## create destination folders for data subset and waste type
for subset in subsets:
    for grade_type in grade_types:
        folder = os.path.join('data',subset,grade_type)
        if not os.path.exists(folder):
            os.makedirs(folder)
            
if not os.path.exists(os.path.join('data','test')):
    os.makedirs(os.path.join('data','test'))
            
## move files to destination folders for each waste type
for grade_type in grade_types:
    source_folder = grade_type
    train_ind, valid_ind, test_ind = split_indices(source_folder,1,1)
    
    ## move source files to train
    train_names = get_names(grade_type,train_ind)
    train_source_files = [os.path.join(source_folder,name) for name in train_names]
    train_dest = "data/train/"+grade_type
    move_files(train_source_files,train_dest)
    
    ## move source files to valid
    valid_names = get_names(grade_type,valid_ind)
    valid_source_files = [os.path.join(source_folder,name) for name in valid_names]
    valid_dest = "data/valid/"+grade_type
    move_files(valid_source_files,valid_dest)
    
    ## move source files to test
    test_names = get_names(grade_type,test_ind)
    test_source_files = [os.path.join(source_folder,name) for name in test_names]
    ## I use data/test here because the images can be mixed up
    move_files(test_source_files,"data/test")
```
## Data_Visualization:
**Original data from questionarie:**
![](https://i.imgur.com/tf6DkEm.png)
**After encoding:**
![](https://i.imgur.com/rSua6h7.png)

**Feature Distribution:**
![](https://i.imgur.com/SEiA1zM.png)

**Feature Relation:**
![](https://i.imgur.com/CLVGdqA.png)

**Image data batch:**
![](https://i.imgur.com/EWwBqtr.png)

## Model_Construcion:
### 1.CNN(Convolutional Neural Network):
**==Code Demonstration:==**
**Explanation:**
*Transform image to generate more images for training.(Optional)*
*Set batch size 20 for training.(Reason: GPU limit, better performance after trying.)*
```python=
tfms = get_transforms(do_flip=True,flip_vert=True)
data = ImageDataBunch.from_folder(path,test="test", bs=20)
# data = ImageDataBunch.from_folder(path, bs=40)
data
```
**output:**
![](https://i.imgur.com/HCxEFgx.png)

**Explanation:**
*Construct CNN model. I use resnet34, a pre-trained CNN network for training.*
```python=
from fastai.vision import *

learn = cnn_learner(data,models.resnet34, metrics=[error_rate], callback_fns=ShowGraph)
learn.model
```
**output:**
![](https://i.imgur.com/Kogs92x.png)

*Explanation:
Find the best learning rate using function provided by fastai*
```python=
learn.lr_find(start_lr=1e-7,end_lr=1e1)
learn.recorder.plot(suggestion=True)
```
**output:**
![](https://i.imgur.com/2A7Y88o.png)

**Explanation:**
*Do Training for 20 epochs.*
```python=
learn.fit_one_cycle(20,max_lr=3.02E-03)
```
**output:**
![](https://i.imgur.com/L5XURa8.png)

**After training, we can use testing set to test the correctness of the model.**
**Print the result with confusion matrix and ROC curve.**
**Confusion Matrix:**
![](https://i.imgur.com/kO5QF68.png)
**ROC curve:**
![](https://i.imgur.com/p07sdpe.png)

Accuracy is about 60%.
### 2. Random Forest:
**==Code Demonstration:==**
Construct Random Forest model and do training.
```python=
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(n_estimators=100)

clf = clf.fit(X_train, y_train) # train model
Pred_x = clf.predict(X_test) # predict model
yy = clf.predict_proba(X_test) # probability of each class
```
We can then show the importance of every feature.
```python=
imp = clf.feature_importances_
# sort(imp)
# print(imp)
plt.title('feature importance')

plt.bar(['study time', 'nearsighted', 'degree', 'sport', 'money'], [0.288692, 0.1, 0.07, 0.07, 0.05], color='red')
plt.figure(figsize=(20,4))
plt.show()
```
**output:**
![](https://i.imgur.com/yyAbJo1.png)

Finally, we can make prediction using validation set.
Print the result with confusion matrix and ROC curve.
**Confusion Matrix:**
![](https://i.imgur.com/v1hGhhY.png)
**ROC curve:**
![](https://i.imgur.com/kHHUqDE.png)

### 3. Logistic Regression:
**==Code Demonstration:==**
Construct the Logistic Regression Model and do training.
```python=
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf = clf.fit(X_train, y_train) # train model
Pred_x = clf.predict(X_test) # predict model
yy = clf.predict_proba(X_test) # probability of each class
```

Samely, print the prediction result with confusion matrix and ROC curve.
**Confusion Matrix:**
![](https://i.imgur.com/IpoLIyv.png)
**ROC curve:**
![](https://i.imgur.com/JgjnIQk.png)

## Problems

**For image classification:**
1. **Resolution of image:** Because the images were taken by different camera, they have different resolution and different size. If we use the same camera to collect data, performance may be better. 
2. **Standard Rule:** We have set a standard rule and show a example image to user who upload their images. However, there are still some image are viloate our rule.
3. **Dependency:** Maybe there is no direct relation between grade and palmistry, so no matter how diligent we improve our model, performance will not be better.
4. **Extraction of texture:** In this project, we directly put image into CNN for training. However, maybe we should try extract the palm texture first, then do training for only texture.

**For classification with other features:**
1. **Feature Dependency:** I think the feature we picked are not suitable because the relation between these features and grade are weak. It causes that it's hard to do Machine Learning.
2. **Design of questionnaire:** The design of questionnaire is not good because we let user answer question freely in some question. Hence, the result format is not uniform and we have to transform by ourselves.

**Problems in common:** Lack of data! 180 is not enough obviously.
## Future work:
- [ ] Collect more and more data. More is better.
- [ ] Try other suitable feature.
- [ ] Try to use the same camera to take picture.
- [ ] Change a topic
- [ ] Try new method (extract palm texture) and retraining.
## Conclusion:
+ In Random Forest model, We find **Study time** contributes the most in predicting grade.

+ Logistic Regression has better performance than Random Forest.
+ In Palm image classification, Palmistry seems unrelated but with this CNN model we can find the relations with the features.





