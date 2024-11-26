require_relative "../test/test_helper"
require_relative "../common/global"
require_relative "sigmoid_layer"

describe SigmoidLayer do
  it "forward and backward for sigmoid layer" do
    x = Torch.tensor(0, dtype: :float64)
    sigmoid_layer = SigmoidLayer.new

    # forward
    assert Torch.equal(Torch.tensor(0.5, dtype: :float64), sigmoid_layer.forward(x))

    # backward
    dout = Torch.tensor(0.5, dtype: :float64)

    assert Torch.equal(Torch.tensor(0.125, dtype: :float64), sigmoid_layer.backward(dout))
  end
end
