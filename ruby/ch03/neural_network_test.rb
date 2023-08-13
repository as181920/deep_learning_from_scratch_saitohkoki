require_relative "../test/test_helper"
require_relative "neural_network"

describe NeuralNetwork do
  it "should calc neural network forward" do
    nn = NeuralNetwork.new
    x = Torch.tensor([1.0, 0.5])
    y = nn.forward(x)

    assert_equal [2], y.shape
    assert_in_delta 0.69627909, y[1].to_f
  end
end
