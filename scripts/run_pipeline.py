import os
import sys
import argparse
import pandas as pd

    
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train import train_model
from src.models.evaluate import evaluate_predictions, print_classification_report
from src.models.optuna_tune import run_optuna
from src.utils.validate_data import basic_schema_checks, sanity_checks
from src.utils.mlflow_utils import setup_mlflow, start_run, log_params, log_metrics, log_artifact


def main():
    parser = argparse.ArgumentParser(description="Pipeline Telco Churn")
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        help="Chemin vers le CSV brut",
    )
    parser.add_argument("--target", type=str, default="Churn", help="Colonne cible")
    parser.add_argument("--model_dir", type=str, default=os.path.join("artifacts", "model"))
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mlflow_uri", type=str, default=None, help="MLflow Tracking URI (optionnel)")
    parser.add_argument("--experiment", type=str, default="churn-ml", help="Nom d'expérience MLflow")
    parser.add_argument("--cv", type=int, default=3, help="Nombre de folds CV pour Optuna")
    parser.add_argument("--optuna_trials", type=int, default=30, help="Nombre d'essais Optuna (défaut: 30)")
    args = parser.parse_args()

    print("1) Chargement des données…")
    df = load_data(args.csv)

    # Validations de base sur le brut
    ok_schema, missing = basic_schema_checks(df, required_columns=["Churn"]) if args.target == "Churn" else (True, [])
    if not ok_schema:
        raise ValueError(f"Colonnes manquantes: {missing}")
    ok_sanity, issues = sanity_checks(df)
    if not ok_sanity:
        print(f"Avertissement validation: {issues}")

    print("2) Prétraitement…")
    df = preprocess_data(df, target_col=args.target)

    print("3) Construction des features…")
    df_feat = build_features(df, target_col=args.target)

    # Initialise MLflow
    setup_mlflow(tracking_uri=args.mlflow_uri, experiment_name=args.experiment)

    print("4) Tuning avec Optuna puis entraînement…")
    with start_run(run_name="baseline_xgb"):
        # Toujours utiliser Optuna par défaut (sans argument requis)
        from src.models.train import split_features_target
        X_all, y_all = split_features_target(df_feat, args.target)
        optuna_out = run_optuna(X_all, y_all, cv=args.cv, n_trials=args.optuna_trials, threshold=args.threshold)
        best_params = optuna_out["best_params"]
        print(f"   Optuna meilleurs hyperparamètres: {best_params}")
        print(f"   Meilleur recall Optuna: {optuna_out['best_value']:.4f}")
        
        # Entraîner avec les meilleurs hyperparams
        train_out = train_model(
            df_feat,
            target_col=args.target,
            test_size=args.test_size,
            model_dir=args.model_dir,
            model_params=best_params,
            threshold=args.threshold,
        )

        print("5) Évaluation…")
        y_true = train_out["y_test"]
        y_proba = train_out["y_proba"]
        train_time = train_out["train_time"]
        pred_time = train_out["pred_time"]
        
        print(f"⏱ Training time: {train_time:.2f} seconds")
        print(f"⏱ Prediction time: {pred_time:.4f} seconds")
        
        metrics = evaluate_predictions(y_true, y_proba, threshold=args.threshold)
        print_classification_report(y_true, y_proba, threshold=args.threshold)

        print("Métriques:")
        for k, v in metrics.items():
            print(f" - {k}: {v:.4f}")

        # Log MLflow
        log_params({
            "model": "xgboost",
            "test_size": args.test_size,
            "threshold": args.threshold,
            "optuna_best_recall": optuna_out["best_value"],
        })
        log_params({f"optuna_{k}": v for k, v in best_params.items()})
        log_metrics(metrics)
        log_metrics({
            "train_time": train_time,
            "pred_time": pred_time,
        })
        log_artifact(os.path.join(args.model_dir, "model.pkl"))
        log_artifact(os.path.join(args.model_dir, "feature_columns.txt"))

    print(f"Modèle sauvegardé dans: {args.model_dir}")


if __name__ == "__main__":
    main()



# Options disponibles :
# --optuna_trials 50 : plus d'essais Optuna
# --threshold 0.25 : seuil personnalisé
# --mlflow_uri mlruns : activer MLflow