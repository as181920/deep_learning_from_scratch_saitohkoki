require_relative "../test/test_helper"
require_relative "neural_network_mnist"
require_relative "../dataset/mnist"

describe NeuralNetworkMnist do
  it "should predict image" do
    nn = NeuralNetworkMnist.new
    image = Mnist.load_train_images(dtype: :float32).first
    y = nn.predict(image)

    assert_in_delta 1, y.sum.to_f
  end

  it "should calculate accuracy" do
    nn = NeuralNetworkMnist.new

    assert nn.calculate_accuracy > 0.9
  end
end
