require_relative "../test/test_helper"
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
    x = Torch.rand(100, 784, dtype: :float64)
    y = net.predict(x)

    assert_equal [100, 10], y.shape
  end

  it "implement numerical_gradient" do
    net = TwoLayerNet.new
    x = Torch.rand(100, 784, dtype: :float64)
    t = Torch.rand(100, 10).argmax(1)
    grads = net.numerical_gradient(x, t)

    assert_equal [784, 100], grads["W1"].shape
    assert_equal [100], grads["b1"].shape
    assert_equal [100, 10], grads["W2"].shape
    assert_equal [10], grads["b2"].shape
  end
end
