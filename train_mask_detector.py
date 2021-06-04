#we start by importing necessary libraries

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
import scikitplot as skplt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



#we set the number of epochs to high number since we will use the Early Stopping Callback

epochs = 100

#we use the current_datetime to name the models in order to make it easier to compare different models
current_datetime = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
dataset_name = "small_dataset"
DIRECTORY = os.getcwd()
dataset_folder_PATH = os.path.join(DIRECTORY, dataset_name)
CATEGORIES = ["with_mask", "without_mask"]


# we import the images from the dataset

print(f"Importing images from {dataset_folder_PATH}.")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(dataset_folder_PATH, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# here we one-hot-encode the labels
        
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

labels = np.array(labels)
images = np.array(data, dtype="float32")


x_train, x_test, y_train, y_test = train_test_split(images, labels,
	test_size=0.30, stratify=labels, random_state=100)


# we now load the MobileNetv2 and we construct the head of the model that will be placed on top of the
# the base model

model_base = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

 
model_head = model_base.output
model_head = AveragePooling2D(pool_size=(7, 7))(model_head)
model_head = Flatten(name="flatten")(model_head)
model_head = Dense(128, activation="relu")(model_head)
model_head = Dropout(0.5)(model_head)
model_head = Dense(2, activation="softmax")(model_head)


final_model = Model(inputs=model_base.input, outputs=model_head)


#now we freeze the layers in the base model to make sure they are not 
#trained  (we only want to  ttrin the head) and we compile the model

for x in model_base.layers:
	x.trainable = False


INIT_LR = 1e-4
BS = 32
opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)
final_model.compile(loss="binary_crossentropy", optimizer="adam",
	metrics=["accuracy"])

val_split = 0.2


#Here we set up the 2 callbacks we''ll ue for training

es = EarlyStopping(monitor='val_accuracy', patience = 2)
checkpoint_filepath = os.path.join(DIRECTORY, "model_{}".format(current_datetime))
mc = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    save_best_only=True)





print(f"\nCurrently training on {round(len(x_train) * (1 - val_split))} samples and validating on {round(len(x_train) * val_split)} samples.")
train_history = final_model.fit(x_train, y_train,
	validation_split = val_split,
	epochs=epochs, callbacks = [es, mc])

final_model = load_model(checkpoint_filepath)

# make predictions on the testing set

print(f"\nEvaluating performance using {len(x_test)} test images.")
preds = final_model.predict(x_test, batch_size=BS)



#here we create the folder where we will store the training evaluation charts 

model_PATH = f"CHARTS_mask_detector_{current_datetime}_totalepochs{len(train_history.history['loss'])}_{dataset_name}"
os.makedirs(model_PATH)



# here we calculate the CLASSIFICATION REPORT

preds = np.argmax(preds, axis=1)
print(classification_report(y_test.argmax(axis=1), preds,
	target_names=lb.classes_))

report = classification_report(y_test.argmax(axis=1), preds,
	target_names=lb.classes_, output_dict = True)
report_df = pd.DataFrame(report).transpose()
report_df.to_excel(os.path.join(model_PATH, f"classification_report_{current_datetime}.xlsx"))



#Here we plot the CONFUSION MATRIX

import seaborn as sn
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test.argmax(axis=1), preds)
df = pd.DataFrame(matrix, index = ["Mask", "No Mask"], columns = ["Mask Predicted", "No Mask Predicted"])
plt.figure(figsize = (10,7))


group_counts = ["{0:0.0f}".format(value) for value in
                matrix.flatten()]
percentages = []
percentages.append(matrix[0,0] / np.sum(matrix.flatten()[0:2]))
percentages.append(matrix[0,1] / np.sum(matrix.flatten()[0:2]))
percentages.append(matrix[1,0] / np.sum(matrix.flatten()[2:]))
percentages.append(matrix[1,1] / np.sum(matrix.flatten()[2:]))
group_percentages = ["{0:.2%}".format(value) for value in
                     percentages]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sn_plot = sn.heatmap(df, annot=labels, fmt="", cmap='Blues')
figure = sn_plot.get_figure()   
figure.savefig(os.path.join(model_PATH, f"confusion_matrix_{current_datetime}.png"), dpi=400)


TP = matrix[0, 0]
FP = matrix[0, 1]
FN = matrix[1, 0]
TN = matrix[1, 1]
Se = TP / (TP + FN)
Sp = TN / (TN + FP)
y_index = Se + Sp - 1



#here we plot the ROC CURVE

from sklearn.metrics import roc_curve


fpr_keras, tpr_keras, thresholds_keras = roc_curve(np.argmax(y_test, axis = 1), preds)

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)
plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label=f"ROC (AUC = {np.round(auc_keras, 2)}, Youden's index = {np.round(y_index, 2)})")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig(os.path.join(model_PATH, f"ROC_curve_{current_datetime}.png"))


#here we plot the CUMULATIVE GAINS CHART and the LIFT CHART

predicted_probas = final_model.predict(x_test)

skplt.metrics.plot_cumulative_gain(np.argmax(y_test, axis = 1), predicted_probas)
plt.show()
figure = skplt.metrics.plot_cumulative_gain(np.argmax(y_test, axis = 1), predicted_probas, "Gains Chart (Class 0: Mask - Class 1: No Mask)").get_figure()
figure.savefig(os.path.join(model_PATH, f"Gains_Chart_{current_datetime}.png"), dpi=400)

skplt.metrics.plot_lift_curve(np.argmax(y_test, axis = 1), predicted_probas)
plt.show()
figure = skplt.metrics.plot_lift_curve(np.argmax(y_test, axis = 1), predicted_probas, "Lift Chart (Class 0: Mask - Class 1: No Mask)").get_figure()
figure.savefig(os.path.join(model_PATH, f"Cumulative_Lift_Chart_{current_datetime}.png"), dpi=400)



#here we plot the accuracy and loss chart

def plot_results(train_history, epochs):
    plt.figure()
    plt.plot(np.arange(0, epochs), train_history.history["accuracy"], label="train_acc", c = "b")
    plt.plot(np.arange(0, epochs), train_history.history["val_accuracy"], label="val_acc", c = "r")
    plt.plot(np.arange(0, epochs), train_history.history["loss"], label="train_loss", c = "b")
    plt.plot(np.arange(0, epochs), train_history.history["val_loss"], label="val_loss", c = "r")
    plt.title("Accuracy and Loss values throughout training")
    plt.ylabel("Accuracy/Loss")
    plt.xlabel("epochs")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(model_PATH, f"train_history_plot_{current_datetime}.png"))


plot_results(train_history, len(train_history.history["loss"]))



print(f"All models saved in the {model_PATH} folder.")

























