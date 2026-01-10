import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import face_recognition

names = []
for filename in os.listdir('interface_face'):
    name, ext = os.path.splitext(filename)
    if ext.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
        names.append(name)

names = sorted(names)

def create_list(csv_file):
    df = pd.read_csv('./labels/labels.csv')
    df_yes = df[df['label']=='yes'] 

    face_id = []
    for face in df_yes['name'].values:
        face_id.append(face.split('-')[1])

    user_yes = []
    for index, name in enumerate(names, start=1):
        if str(index) in face_id:
            user_yes.append(name)

    return user_yes


#faces from faces.pkl 
def data_preprocess(yes_list, pickle_file):
    with open(pickle_file, 'rb') as f:
        faces = pickle.load(f)

    rows = []
    for person, encs in faces.items():
        for enc in encs:
            row = {'name': person}

            for i, value in enumerate(enc):
                row[f'f{i}'] = value

            rows.append(row)

    df = pd.DataFrame(rows)


    df['target'] = df['name'].isin(yes_list).astype(bool)

    X = df.filter(regex="^f").values 
    X = normalize(X, norm='l2')
 
    # binary target 
    y = df['target'].astype(int).to_numpy() # 1 = True, 0 = False 

    # group by indentity to avoid leakage 
    groups = df['name'].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
 
    return X_train, X_test, y_train, y_test 



def run_model(csv_file, pickle_file, test_file):
    yes_list = create_list(csv_file)

    X_train, X_test, y_train, y_test = data_preprocess(yes_list, pickle_file)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",  # helpful if yop_yes is smaller
        solver="lbfgs"
    )

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, pred))
    print("ROC AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, pred, digits=4))


    image = face_recognition.load_image_file(test_file)
    test = face_recognition.face_encodings(image)

    X_new = np.vstack(test)
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    X_new = X_new / (norms + 1e-12)

    proba = clf.predict_proba(X_new)[:,1]   
    test_name = test_file.split('/')[-1].split('.')[0]
    return f"the probability of you falling in love with {test_name} is {proba[0]}."  


print(run_model("./labels/labels.csv", "./faces.pkl","./testing/sarp2.png"))