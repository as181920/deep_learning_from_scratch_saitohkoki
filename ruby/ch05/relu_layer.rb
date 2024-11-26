require "torch-rb"
require_relative "../common/global"

class ReluLayer
  attr_reader :mask

  def forward(x)
    @mask = x.lt(0)
    x.clamp(0)
  end

  def backward(dout)
    dout[mask] = 0
    dout
  end
end
