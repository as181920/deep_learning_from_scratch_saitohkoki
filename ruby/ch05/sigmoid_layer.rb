require "torch-rb"
require_relative "../common/global"

class SigmoidLayer
  attr_reader :out

  # x: Torch.tensor
  def forward(x)
    @out = 1 / (1 + Torch.exp(x))
    out
  end

  def backward(dout)
    dout * out * (1 - out)
  end
end
