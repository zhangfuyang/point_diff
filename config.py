from dataclasses import dataclass

@dataclass
class TrainingConfig:
    train_batch_size = 32
    eval_batch_size = 4
    #room_num = 2 
    num_epochs = 200
    learning_rate = 5e-4
    output_dir = "pointset" 
    seed = 1234
    data_path = 'data_bank'
