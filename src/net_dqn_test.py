import torch
from breakout_game import Breakout
from utils import device, preprocessing
from models import NeuralNetwork


def test(model):
    game_state = Breakout()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32).to(device)
    action[0] = 1
    image_data, _, _ = game_state.take_action(action)
    image_data = preprocessing(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # get output from the neural network
        output = model(state)[0]
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)

        # get action
        action_index = torch.argmax(output)
        action[action_index] = 1

        # get next state
        image_data_1, _, _ = game_state.take_action(action)
        image_data_1 = preprocessing(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1     


if __name__ == "__main__":
    model = NeuralNetwork() 
    model.load_state_dict(torch.load('model_weights/best.pth').state_dict())
    test(model.eval().to(device))
