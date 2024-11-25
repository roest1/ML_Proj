# Cross validation example

---

This is used for a regression case so you just need to switch the accruacy metrics from r2 and mse to things like accuracy, precision, f1, and recall.

```python

def cross_validation(dataframe: pd.DataFrame, input_columns: list, target_columns: list, k_folds=5):
    """
    Perform k-fold cross-validation, ensuring that trials are split according to
    unique health and shoe conditions, utilizing the existing train_test_split function.
    """
    unique_trials = dataframe['Trial'].unique()
    gss = GroupShuffleSplit(
        n_splits=k_folds, test_size=1/k_folds, random_state=42)

    r2_scores, mse_scores = [], []

    for fold, (train_val_idx, test_idx) in enumerate(gss.split(unique_trials, groups=unique_trials), 1):
        print(f"Processing fold {fold}...")
        train_val_trials = unique_trials[train_val_idx]
        test_trials = unique_trials[test_idx]

        # Generate a mask for selecting the dataframe rows corresponding to the current fold's train and test trials
        train_val_mask = dataframe['Trial'].isin(train_val_trials)
        test_mask = dataframe['Trial'].isin(test_trials)

        # Split the dataframe into training/validation and testing dataframes based on the mask
        train_val_df = dataframe[train_val_mask]
        test_df = dataframe[test_mask]

        X_train, y_train, X_val, y_val, X_test, y_test, _, _ = train_test_split(
            dataframe=train_val_df,
            input_columns=input_columns,
            target_columns=target_columns,
            verbose=False
        )

        X_test, y_test = torch.tensor(test_df[input_columns].values, dtype=torch.float32), torch.tensor(
            test_df[target_columns].values, dtype=torch.float32)

        # Initialize and train the model
        model = NeuralNet(X_train.shape[1], y_train.shape[1])
        trained_model = train_neural_network(
            model, X_train, y_train, X_val, y_val, verbose=False)

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            y_pred = trained_model(X_test)
            r2_scores.append(r2_score(y_test.numpy(), y_pred.numpy()))
            mse_scores.append(mean_squared_error(
                y_test.numpy(), y_pred.numpy()))

        print(
            f"Fold {fold}: R² = {r2_scores[-1]:.4f}, MSE = {mse_scores[-1]:.4f}")
    print("="*50)
    print(
        f"\nAverage R²: {np.mean(r2_scores):.4f}, Average MSE: {np.mean(mse_scores):.4f}")

```
