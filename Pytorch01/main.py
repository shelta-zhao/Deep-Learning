# @ Time : 2022/3/21,10:23
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description: Predict the positive cases based on the data of 5 days ago


import pandas as pd
from Dataset import *
from Model import *
from torch.utils.data import DataLoader

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'seed': 1127,  # Your seed number, you can pick your lucky number. :)
        'select_all': True,  # Whether to use all features.
        'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
        'n_epochs': 10,  # Number of epochs.
        'batch_size': 256,
        'learning_rate': 1e-5,
        'early_stop': 400,  # If model has not improved for this many consecutive epochs, stop training.
        'save_path': './models/model.ckpt'  # Your model will be saved here.
    }
    # Set seed for reproducibility
    same_seed(config['seed'])

    # train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days)
    # test_data size: 1078 x 117 (without last day's positive rate)
    train_data, test_data = pd.read_csv('./data/covid.train.csv').values, pd.read_csv('./data/covid.test.csv').values
    train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

    # Print out the data size.
    print(f"""train_data size: {train_data.shape} 
    valid_data size: {valid_data.shape} 
    test_data size: {test_data.shape}""")

    # Select features
    x_train, x_valid, x_test, y_train, y_valid = select_feature(train_data, valid_data, test_data, config['select_all'])

    # Print out the number of features.
    print(f'number of features: {x_train.shape[1]}')

    train_dataset, valid_dataset, test_dataset = COVIDDataset(x_train, y_train), \
                                                 COVIDDataset(x_valid, y_valid), \
                                                 COVIDDataset(x_test)

    # Pytorch data loader loads pytorch dataset into batches.
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

model = MyModel(input_dim=x_train.shape[1]).to(device)  # put your model and data on the same computation device.
trainer(train_loader, valid_loader, model, config, device)
