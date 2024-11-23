require_relative "../test/test_helper"
require_relative "simple_net"

describe SimpleNet do
  it "implement predict and loss" do
    net = SimpleNet.new
    x = Torch.tensor([0.6, 0.9], dtype: :float64)
    p = net.predict(x)

    assert_equal [3], p.shape

    t = Torch.tensor([0, 0, 1])

    assert_operator net.loss(x, t), :>, 0
  end
end
