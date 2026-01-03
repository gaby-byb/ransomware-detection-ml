
import argparse
import joblib
import pandas as pd


def detect(model_path, data_path, labels_path=None, threshold=0.5):

    #Load model
    model = joblib.load(model_path)
    
    #features used for prediction
    X = pd.read_csv(data_path)
    #Label used for confirmation 
    y = pd.read_csv(labels_path)

    test_sample = X.sample(n=1, random_state=None)
    test_sample_index = test_sample.index[0]
    print("Random sample selected for detection.\n")
    
    #holds the target feature of the sample
    true_label = y.iloc[test_sample_index].values[0]
    if true_label == 0:
        real_label = "Ransomware"
    else:
        real_label = "Benign"


    #predict probability of ransomware
    prob = model.predict_proba(test_sample)[0][1]
    label = "Benign" if prob > 0.6 else "Ransomware"

    print(f"Model prediction: {label}")
    print(f"True Label: {real_label}.\n")

    if str(real_label) == label:
        print("Model predictect correctly")
    else:
        print("Model failed")


def main():
    parser = argparse.ArgumentParser(
         description="Universal Malware Detection CLI Tool"
     )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to the trained model (.pkl file)"
    )
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to the processed dataset (.csv file)"
    )
    parser.add_argument(
        "--labels", "-l",
        required=True,
        help="Path to the labels file (.csv file)"
        )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Probability threshold for ransomware classification (default=0.5)"
    )

    args = parser.parse_args()
    #it calls the detect function
    detect(args.model, args.data, args.labels, args.threshold)
#command line
    
if __name__ == "__main__":
    main()
