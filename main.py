import pandas as pd
import missingno as msno

def main(): 
    df = pd.read_csv('./data/data.csv')
    print(df.info())
    

if __name__ == '__main__': 
    main()