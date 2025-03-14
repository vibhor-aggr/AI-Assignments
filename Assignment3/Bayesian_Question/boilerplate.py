#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model
# import time

######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    # Example: train_data = pd.read_csv("train_data.csv")
    train_df = pd.read_csv("train_data.csv")
    val_df = pd.read_csv("validation_data.csv")
    return train_df, val_df

def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model
    DAG = bn.make_DAG([
         ('Start_Stop_ID', 'Zones_Crossed'),
         ('End_Stop_ID', 'Zones_Crossed'),
         ('Zones_Crossed', 'Distance'),
         ('Route_Type', 'Fare_Category'),
         ('Distance', 'Fare_Category')
         ])
    model = bn.parameter_learning.fit(DAG, df, methodtype='maximumlikelihood')
    return model

def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""
    # Code to create a pruned network, fit it, and return the pruned model
    DAG = bn.make_DAG([
         ('Distance', 'Fare_Category')
         ])
    model = bn.parameter_learning.fit(DAG, df, methodtype='maximumlikelihood')
    return model

def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # Code to optimize the structure, fit it, and return the optimized model
    optimized_model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
    optimized_model = bn.parameter_learning.fit(optimized_model, df, methodtype='maximumlikelihood')
    return optimized_model

def save_model(fname, model):
    """Save the model to a file using pickle."""
    with open(fname, "wb") as f:
        pickle.dump(model, f)

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

############
## Driver ##
############

def main():
    # Load data
    # start_time = time.time()
    train_df, val_df = load_data()
    # end_time = time.time()
    # runtime = end_time - start_time
    # print(f"Runtime for loading datasets: {runtime} s")

    # Create and save base model
    # start_time = time.time()
    base_model = make_network(train_df.copy())
    save_model("base_model.pkl", base_model)
    # end_time = time.time()
    # runtime = end_time - start_time
    # print(f"Runtime for initial Bayesian network: {runtime} s")
    # bn.plot(base_model)

    # Create and save pruned model
    # start_time = time.time()
    pruned_network = make_pruned_network(train_df.copy())
    save_model("pruned_model.pkl", pruned_network)
    # end_time = time.time()
    # runtime = end_time - start_time
    # print(f"Runtime for pruned Bayesian network: {runtime} s")
    # bn.plot(pruned_network)

    # Create and save optimized model
    # start_time = time.time()
    optimized_network = make_optimized_network(train_df.copy())
    save_model("optimized_model.pkl", optimized_network)
    # end_time = time.time()
    # runtime = end_time - start_time
    # print(f"Runtime for optimized Bayesian network: {runtime} s")
    # bn.plot(optimized_network)

    # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    print("[+] Done")

if __name__ == "__main__":
    main()

