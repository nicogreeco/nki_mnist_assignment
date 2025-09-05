from argparse import ArgumentParser
import os
import joblib
from sklearn.linear_model import LogisticRegression
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from utils import embed_val_given_ckpt_path

def main(args):
    X, y = embed_val_given_ckpt_path(args.ckpt_path, args.model)

    clf = LogisticRegression(
        random_state=0,
        n_jobs=4,
        max_iter=1000,          # <- often needed for convergence
    ).fit(X, y)

    
    y_pred = clf.predict(X)
    
    print(f"Accuracy on validation set: {(y_pred == y).sum()/len(y)}")
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    joblib.dump(clf, args.out_path)   # e.g., "models/logreg.joblib"
    print(f"Saved classifier to: {args.out_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="log/lightning_logs/version_1/checkpoints/best-epoch=80-val_loss=0.0600.ckpt")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="cnn")
    
    parser.add_argument(
        "--out_path", 
        type=str, 
        default="log/logistic_regression/log_reg.ckpt")

    args = parser.parse_args()
    main(args)