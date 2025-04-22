from MNISTData import MNISTData
from AutoEncoder import AutoEncoder
import os 

if __name__=="__main__":
    print("Hi. I'm an Auto Encoder Trainer.")
    batch_size = 32
    num_epochs = 5

    data_loader = MNISTData()
    data_loader.load_data()

    x_train = data_loader.x_train
    input_output_dim = data_loader.in_out_dim

    auto_encoder = AutoEncoder()
    auto_encoder.build_CNN_model()
    auto_encoder.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=num_epochs)

    # Ensure the directory exists
    save_dir = "./model"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "ae_model.weights.h5")
    auto_encoder.save_weights(save_path)
    print("load model weights from %s" % save_path)