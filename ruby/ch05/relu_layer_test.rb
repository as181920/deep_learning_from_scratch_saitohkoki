require_relative "../test/test_helper"
require_relative "../common/global"
require_relative "relu_layer"

describe ReluLayer do
  it "forward and backward for relu layer" do
    x = Torch.tensor([[-1, 0, 1], [-2, 0, 2]])

    relu_layer = ReluLayer.new

    # forward
    assert Torch.equal(Torch.tensor([[0, 0, 1], [0, 0, 2]]), relu_layer.forward(x))
    assert Torch.equal(Torch.tensor([[true, false, false], [true, false, false]]), relu_layer.mask)

    # backward
    dout = Torch.tensor([[-1, 0, 1], [-2, 0, 2]])

    assert Torch.equal(Torch.tensor([[0, 0, 1], [0, 0, 2]]), relu_layer.backward(dout))
  end
end
