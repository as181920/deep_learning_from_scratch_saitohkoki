require_relative "../test/test_helper"
require_relative "neural_network"

describe NeuralNetwork do
  it "should implement step function" do
    assert_equal 0, NeuralNetwork.step_function_for_num(-1)
    assert_equal 1, NeuralNetwork.step_function_for_num(999)
  end

  it "should implement step function for tensor" do
    assert Torch.equal(
      Torch.tensor([0, 0, 0, 1, 1]),
      NeuralNetwork.step_function(Torch.tensor([-99, -1, 0, 1, 999]))
    )
  end

  it "should implement sigmoid function" do
    x = Torch.tensor([-99, -2, 0, 1, 123])

    assert Torch.equal(Torch.sigmoid(x), NeuralNetwork.sigmoid(x))
  end

  it "should implement relu function" do
    x = Torch.tensor([-99, -2, 0, 1, 123])

    assert Torch.equal(Torch.tensor([0, 0, 0, 1, 123]), NeuralNetwork.relu(x))
  end
end
