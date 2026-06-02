from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 
  
# metadata 
# print(statlog_german_credit_data.metadata) 
  
# variable information 
# print(statlog_german_credit_data.variables)

# function to load data
def load_data():
    # concat data and target
    data = X.copy()
    data['target'] = y
    
    return data

# function to save data to csv
def save_data(data, file_path):
    data.to_csv(file_path, index=False)

if __name__ == "__main__":
    print(X.shape)
    print(y.shape)