import matplotlib.pyplot as plt
from predict_model import *
from make_dataset import *

xgb.plot_tree(load_model, rankdir='LR')
fig = plt.gcf()
fig.set_size_inches(150, 100)
plt.savefig('figure1.png', bbox_inches='tight')
plt.show()

im_features = load_model.feature_importances_
plt.figure(figsize=(12,5))
plt.bar(range(len(im_features)), im_features)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importances")
plt.xticks(range(len(im_features)), X.columns.to_list(), rotation=45)
plt.savefig('figure2.png', bbox_inches='tight')
plt.show()
