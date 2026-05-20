from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATASET_ROOT=Path("/home/caio-torkst/projects/tcc/IPqM-Fall/organized")
WINDOWS_CSV="window_labels.csv"

FEATURE_COLUMNS=["ax","ay","az","amag","wx","wy","wz","wmag"]

FALL_CLASSES={"FALL_1","FALL_2","FALL_3","FALL_5","FALL_6"}

RESULTS_FILE="results.json"

CHECKPOINT_DIR=Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

if Path(RESULTS_FILE).exists():
    with open(RESULTS_FILE,"r") as f:
        results=json.load(f)
else:
    results={}

windows_df=pd.read_csv(WINDOWS_CSV)

cache={}
X=[]
y=[]
groups=[]
sensor_positions=[]

for _,row in tqdm(windows_df.iterrows(),total=len(windows_df),desc="Loading windows"):
    parquet_path=DATASET_ROOT/row["file"]

    if parquet_path not in cache:
        cache[parquet_path]=pd.read_parquet(parquet_path)

    df=cache[parquet_path]

    window=df.iloc[row["start_idx"]:row["end_idx"]]

    features=window[FEATURE_COLUMNS].to_numpy().flatten()

    label="FALL" if row["label"] in FALL_CLASSES else "NON_FALL"

    X.append(features)
    y.append(label)
    groups.append(row["subject_id"])
    sensor_positions.append(row["sensor_pos"])

X=np.array(X)
y=np.array(y)
groups=np.array(groups)
sensor_positions=np.array(sensor_positions)

logo=LeaveOneGroupOut()

for sensor in ["CHEST","LEFT","RIGHT"]:
    idx=sensor_positions==sensor

    X_sensor=X[idx]
    y_sensor=y[idx]
    groups_sensor=groups[idx]

    folds=list(logo.split(X_sensor,y_sensor,groups_sensor))

    if sensor not in results:
        results[sensor]={
            "folds":[]
        }

    completed_folds=set(
        fold["fold"]
        for fold in results[sensor]["folds"]
    )

    pbar=tqdm(folds,desc=sensor,total=len(folds))

    for fold_number,(train_idx,test_idx) in enumerate(pbar,start=1):
        if fold_number in completed_folds:
            pbar.set_postfix_str(f"fold={fold_number} skipped")
            continue

        X_train,X_test=X_sensor[train_idx],X_sensor[test_idx]
        y_train,y_test=y_sensor[train_idx],y_sensor[test_idx]

        test_subject=groups_sensor[test_idx][0]

        checkpoint_path=CHECKPOINT_DIR/f"{sensor}_fold_{fold_number}_subject_{test_subject}.joblib"

        pbar.set_postfix_str(f"fold={fold_number} subject={test_subject}")

        if checkpoint_path.exists():
            model=joblib.load(checkpoint_path)
        else:
            model=make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000
                )
            )

            model.fit(X_train,y_train)

            joblib.dump(model,checkpoint_path)

        y_pred=model.predict(X_test)

        acc=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred,pos_label="FALL")
        recall=recall_score(y_test,y_pred,pos_label="FALL")
        f1=f1_score(y_test,y_pred,pos_label="FALL")

        tn,fp,fn,tp=confusion_matrix(
            y_test,
            y_pred,
            labels=["NON_FALL","FALL"]
        ).ravel()

        pbar.set_postfix_str(
            f"fold={fold_number} acc={acc:.4f} f1={f1:.4f}"
        )

        fold_result={
            "fold":fold_number,
            "test_subject":str(test_subject),
            "checkpoint":str(checkpoint_path),
            "accuracy":float(acc),
            "precision":float(precision),
            "recall":float(recall),
            "f1":float(f1),
            "tp":int(tp),
            "fp":int(fp),
            "fn":int(fn),
            "tn":int(tn)
        }

        results[sensor]["folds"].append(fold_result)

        folds_results=results[sensor]["folds"]

        results[sensor]["accuracy_mean"]=float(np.mean([x["accuracy"] for x in folds_results]))
        results[sensor]["accuracy_std"]=float(np.std([x["accuracy"] for x in folds_results]))

        results[sensor]["precision_mean"]=float(np.mean([x["precision"] for x in folds_results]))
        results[sensor]["precision_std"]=float(np.std([x["precision"] for x in folds_results]))

        results[sensor]["recall_mean"]=float(np.mean([x["recall"] for x in folds_results]))
        results[sensor]["recall_std"]=float(np.std([x["recall"] for x in folds_results]))

        results[sensor]["f1_mean"]=float(np.mean([x["f1"] for x in folds_results]))
        results[sensor]["f1_std"]=float(np.std([x["f1"] for x in folds_results]))

        results[sensor]["tp_mean"]=float(np.mean([x["tp"] for x in folds_results]))
        results[sensor]["tp_std"]=float(np.std([x["tp"] for x in folds_results]))

        results[sensor]["fp_mean"]=float(np.mean([x["fp"] for x in folds_results]))
        results[sensor]["fp_std"]=float(np.std([x["fp"] for x in folds_results]))

        results[sensor]["fn_mean"]=float(np.mean([x["fn"] for x in folds_results]))
        results[sensor]["fn_std"]=float(np.std([x["fn"] for x in folds_results]))

        results[sensor]["tn_mean"]=float(np.mean([x["tn"] for x in folds_results]))
        results[sensor]["tn_std"]=float(np.std([x["tn"] for x in folds_results]))

        with open(RESULTS_FILE,"w") as f:
            json.dump(results,f,indent=4)

print(json.dumps(results,indent=4))