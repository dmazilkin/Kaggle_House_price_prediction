from argparse import ArgumentParser

from src.utils import read_data, save_data
from src.clean_data import clean_raw_data
from src.features import add_features_and_split
from src.train import train_model
from src.evaluate import evaluate_model

parser = ArgumentParser()
parser.add_argument('-d', '--data', required=True, help='Provide path to data with flag -d or --data')
parser.add_argument('-m', '--model', required=True, help='Provide type of model')

def main(): 
    args = vars(parser.parse_args())
    df = read_data(args['data'])
    
    data_clean = clean_raw_data(df)
    save_data(data_clean, 'clean_data')
    
    X_train, y_train, X_test, y_test = add_features_and_split(data_clean)
    model = train_model(args['model'], X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    print(f"Model: {args['model']}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")

if __name__ == '__main__': 
    main()
