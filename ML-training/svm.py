import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def quality_metrics(y_true, y_pred, THRESHOLD = 0.5):
  ##### METRICS #####
  # Area under ROC curve
  try:
    AUC = metrics.roc_auc_score(y_true,y_pred)
  except Exception as e:
    print('All values predicited one class!')
    AUC = 0

  # Discrete metrics
  y_pred = y_pred >= THRESHOLD

  TP = np.logical_and(y_pred,y_true).sum()
  TN = np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)).sum()
  FP = np.logical_and(y_pred, np.logical_not(y_true)).sum()
  FN = np.logical_and(np.logical_not(y_pred), y_true).sum()

  # According to definition:
  accuracy = (TP+TN)/len(y_pred)
  sensitivity = TP/(TP+FN)
  specificity = TN/(TN+FP)
  #accuracy = (sensitivity+specificity)/2

  return AUC, accuracy, sensitivity, specificity



bf = cv2.imread("/home/maiki/Downloads/bf.JPG", cv2.IMREAD_GRAYSCALE)
pi_m = cv2.imread("/home/maiki/Downloads/pi_m.JPG", cv2.IMREAD_GRAYSCALE)
pi_m_3 = cv2.imread("/home/maiki/Downloads/pi_m_3.JPG", cv2.IMREAD_GRAYSCALE)


bf_neg = 255-bf

trainingData = np.float32(bf_neg.reshape((-1,1)))
labels = np.round(pi_m/255).reshape((-1,1))
trainingData = trainingData/max(trainingData)

x_tr, x_tst, y_tr, y_tst = train_test_split(trainingData, labels, test_size=0.2, random_state=23)

print(x_tr.shape)
print(y_tr[:].shape)
print(x_tst.shape)
print(y_tst[:].shape)


lr = LogisticRegression(max_iter=1000)

lr.fit(x_tr, y_tr)

y_pred = lr.predict_proba(x_tst)[::,1]

print(quality_metrics(y_tst, y_pred))

# print(z.shape)

'''
res = np.concatenate((bf, pi_m, pi_m_3), axis = 1)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimension
cv2.imshow('output', res)
cv2.waitKey(0)
'''