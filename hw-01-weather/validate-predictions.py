import pandas as pd


def validate_model_predictions(file_path):
    try:
        with open(file_path, 'r') as f:
            predictions = [line.strip() for line in f.readlines()]

            # Check if there is one number per row
            if len(predictions) != len([p for p in predictions if p.replace('.', '', 1).isdigit()]):
                print("Error: Not all rows contain a single number.")
                return False

            # Check if the number of rows matches the test data
            test_data = pd.read_csv('data/test-data.csv')
            if len(predictions) != len(test_data):
                print(f"Error: Number of rows ({len(predictions)}) does not match test data ({len(test_data)}).")
                return False

            # Check if all numbers are integers
            if len([p for p in predictions if p.isdigit()]) != len(predictions):
                print("Error: Not all numbers are integers.")
                return False

            # Check if all numbers are between 1 and 12 (inclusive)
            if len([p for p in predictions if 1 <= int(p) <= 12]) != len(predictions):
                print(
                    f"Error: Some numbers are not between 1 and 12 ({[int(p) for p in predictions if not 1 <= int(p) <= 12]}).")
                return False

            print("Validation successful!")
            return True

    except FileNotFoundError:
        print("File not found.")
        return False


validate_model_predictions('data/model-predictions.csv')