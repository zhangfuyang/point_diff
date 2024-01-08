from dataclasses import dataclass

@dataclass
class TrainingConfig:
    train_batch_size = 48
    eval_batch_size = 4
    #room_num = 2 
    num_epochs = 220
    learning_rate = 6e-4
    output_dir = "log/pointset" 
    seed = 1234
    data_path = 'data_bank'
