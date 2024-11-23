require_relative "../test/test_helper"
require_relative "../common/global"
require_relative "two_layer_net"

describe TwoLayerNet do
  it "initial params" do
    net = TwoLayerNet.new

    assert_predicate net.params, :present?
    assert_equal [784, 100], net.params["W1"].shape
    assert_equal [100], net.params["b1"].shape
    assert_equal [100, 10], net.params["W2"].shape
    assert_equal [10], net.params["b2"].shape
  end

  it "implement predict" do
    net = TwoLayerNet.new
    x = Torch.rand(100, 784, dtype: :float64, device: Global::DEVICE)
    y = net.predict(x)

    assert_equal [100, 10], y.shape
  end

  it "implement numerical_gradient" do
    net = TwoLayerNet.new(input_size: 12, hidden_size: 6, output_size: 2)
    x = Torch.rand(10, 12, dtype: :float64, device: Global::DEVICE)
    t = Torch.rand(10, 2).argmax(1)
    grads = net.numerical_gradient(x, t)

    assert_equal [12, 6], grads["W1"].shape
    assert_equal [6], grads["b1"].shape
    assert_equal [6, 2], grads["W2"].shape
    assert_equal [2], grads["b2"].shape
  end
end
